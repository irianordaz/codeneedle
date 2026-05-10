#!/usr/bin/env python3
"""Benchmark suite orchestrator for LMStudio models.

Discovers all models via `lms ls`, creates per-model config TOMLs from
templates in `configs/models/`, runs benchmarks sequentially with 30s
pauses, produces a Markdown results table, and generates visual reports.

Usage:
    pixi run python benchmark_suite.py --corpus http_server
    pixi run python benchmark_suite.py --corpus jquery --max-tokens 128000
    pixi run python benchmark_suite.py --corpus jquery --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 65000
DEFAULT_CORPORA_DIR = Path("configs/corpora")
DEFAULT_USER_MODELS_DIR = Path("configs/models/user")
DEFAULT_TEMPLATE_DIR = Path("configs/models")
DEFAULT_SAVE_TABLE = Path("results_table.md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Remove special characters for fuzzy matching (lowercase, no punctuation)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def round_to_nearest_thousands(value: int) -> int:
    """Round to nearest thousand."""
    return round(value / 1000) * 1000


def format_max_tokens(value: int) -> str:
    """Format max_tokens as 65K, 128K, etc."""
    rounded = round_to_nearest_thousands(value)
    return f"{rounded // 1000}K"


def sanitize_model_name(name: str) -> str:
    """Replace slashes and special chars with underscores for filenames."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def determine_framework(model_name: str) -> str:
    """Determine framework (gguf or mlx) from the model name."""
    if "-mlx-" in model_name.lower() or "mlx" in model_name.lower():
        return "mlx"
    return "gguf"


def find_best_template(model_name: str, template_dir: Path) -> Path | None:
    """Find the best template TOML for a given LMStudio model name.

    Strategy (in priority order):
    1. Exact match (after normalization).
    2. Prefix match: model name starts with (or contains) the template stem,
       or template stem starts with (or contains) the model name.
    3. Fallback: longest common substring (least preferred).
    4. Among ties, prefer `mlx` templates (indicated by "mlx" in template stem).
    """
    # Strip the -lms-* suffix to get the core model name
    core_name = re.sub(r"-lms-.*$", "", model_name)

    if not template_dir.is_dir():
        return None

    best_match: Path | None = None
    best_score = -1
    best_exact = False

    for toml_file in template_dir.glob("*.toml"):
        stem = toml_file.stem
        norm_core = normalize_name(core_name)
        norm_stem = normalize_name(stem)

        # Priority 1: Exact match (after normalization, ignoring case)
        if norm_core == norm_stem:
            return toml_file  # Exact match, no need to search further

        # Priority 2: Prefix/containment match
        is_prefix_match = (
            norm_core.startswith(norm_stem)
            or norm_stem.startswith(norm_core)
            or norm_core in norm_stem
            or norm_stem in norm_core
        )

        # Priority 3: Longest common substring (least preferred)
        common = _longest_common_substring(norm_core, norm_stem)
        score = len(common)

        # Determine if this is a "better" match than current best
        is_better = False

        if is_prefix_match and not best_exact:
            # Prefix match beats LCS
            if best_match is None or not (
                normalize_name(best_match.stem).startswith(norm_core)
                or norm_core.startswith(normalize_name(best_match.stem))
                or normalize_name(best_match.stem) in norm_core
                or norm_core in normalize_name(best_match.stem)
            ):
                is_better = True
            elif score > best_score:
                is_better = True
        elif score > best_score:
            is_better = True

        if is_better:
            best_match = toml_file
            best_score = score
            best_exact = is_prefix_match

    return best_match


def _longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings."""
    if not s1 or not s2:
        return ""

    m, n = len(s1), len(s2)
    max_len = 0
    end_idx = 0  # ending index in s1

    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i
            else:
                dp[i][j] = 0

    return s1[end_idx - max_len : end_idx]


def update_max_tokens_in_toml(content: str, new_max_tokens: int) -> str:
    """Update the max_tokens value in a TOML file content string.

    The TOML field must remain a numeric value (not formatted as 65K).
    """
    # Keep the numeric value for the TOML field
    return re.sub(
        r"(max_tokens\s*=\s*)\d+",
        rf"\g<1>{new_max_tokens}",
        content,
    )


def create_minimal_toml(
    model_name: str, framework: str, max_tokens: int
) -> str:
    """Generate a minimal TOML config when no template is found."""
    formatted = format_max_tokens(max_tokens)
    return (
        f'name = "{model_name}"\n'
        f'base_url = "http://localhost:1234"\n'
        f"temperature = 0.0\n"
        f"max_tokens = {formatted}\n"
        f"timeout = 600.0\n"
        f"suppress_thinking = true\n"
    )


def read_toml_field(content: str, field: str) -> str | None:
    """Extract the value of a field from a TOML file content string."""
    match = re.search(
        rf"^{field}\s*=\s*[\"']?([^\"'\n#]+)", content, re.MULTILINE
    )
    if match:
        return match.group(1).strip().strip('"').strip("'")
    return None


def parse_lms_ls(output: str) -> list[str]:
    """Parse the output of `lms ls` to extract model names.

    Handles LMStudio's tabular output format. Filters out embedding models.
    """
    models = []
    in_embedding_section = False

    for line in output.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip the summary line
        if stripped.startswith("You have") or stripped.startswith(
            "models, taking"
        ):
            continue

        # Detect section headers (LLM / EMBEDDING) — with or without trailing words
        if re.match(r"^(LLM|EMBEDDING)\b", stripped, re.IGNORECASE):
            in_embedding_section = stripped.upper().startswith("EMBEDDING")
            continue

        # Skip header lines (LLM/EMBEDDING PARAMS...)
        if re.match(r"^(LLM|EMBEDDING)\s+\w+\s+", stripped, re.IGNORECASE):
            continue

        # Skip embedding models entirely
        if in_embedding_section:
            continue

        # Extract model name from the first column
        # Format: "model/name (1 variant)" or "model/name"
        match = re.match(r"^(\S+?)(?:\s*\([^)]*\))?\s+", stripped)
        if match:
            models.append(match.group(1))

    return models


def run_subprocess(
    cmd: list[str], check: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark suite orchestrator for LMStudio models.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to a Python source file (or corpus config). Passed as {code} to bench.py run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens applied to all user configs and CLI overrides (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--corpora-dir",
        type=Path,
        default=DEFAULT_CORPORA_DIR,
        help=f"Directory containing corpus TOMLs (default: {DEFAULT_CORPORA_DIR}).",
    )
    parser.add_argument(
        "--user-models-dir",
        type=Path,
        default=DEFAULT_USER_MODELS_DIR,
        help=f"Directory to write user model TOMLs (default: {DEFAULT_USER_MODELS_DIR}).",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help=f"Directory containing template TOMLs (default: {DEFAULT_TEMPLATE_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without executing benchmarks.",
    )
    parser.add_argument(
        "--save-table",
        type=Path,
        default=DEFAULT_SAVE_TABLE,
        help=f"Path to write the Markdown results table (default: {DEFAULT_SAVE_TABLE}).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def discover_models() -> list[str]:
    """Discover all models via `lms ls`."""
    try:
        result = run_subprocess(["lms", "ls"])
    except FileNotFoundError:
        print(
            "ERROR: 'lms' executable not found. Is LMStudio installed and on PATH?",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode != 0:
        print(
            f"ERROR: 'lms ls' failed with code {result.returncode}: {result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)

    models = parse_lms_ls(result.stdout)
    if not models:
        print(
            "ERROR: No models found via 'lms ls'. Load a model in LM Studio first.",
            file=sys.stderr,
        )
        sys.exit(1)

    return models


def generate_user_configs(
    models: list[str],
    template_dir: Path,
    user_models_dir: Path,
    max_tokens: int,
    dry_run: bool = False,
) -> list[Path]:
    """Create a user config TOML for each model.

    Returns the list of created config file paths.
    """
    user_models_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    for model_name in models:
        framework = determine_framework(model_name)
        formatted_max_tokens = format_max_tokens(max_tokens)
        safe_name = sanitize_model_name(model_name)

        toml_filename = f"{safe_name}-{framework}-{formatted_max_tokens}.toml"
        toml_path = user_models_dir / toml_filename

        if dry_run:
            print(f"  [DRY-RUN] Would create: {toml_path}")
            continue

        # Find best template
        template = find_best_template(model_name, template_dir)

        if template is not None:
            content = template.read_text()
            content = update_max_tokens_in_toml(content, max_tokens)
        else:
            print(
                f"  (no template found for {model_name}, generating minimal config)"
            )
            content = create_minimal_toml(model_name, framework, max_tokens)

        toml_path.write_text(content)
        created.append(toml_path)
        print(f"  Created: {toml_path}")

    return created


def run_benchmarks(
    config_paths: list[Path],
    corpus: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run benchmarks for each config, sleeping 30s between runs.

    Returns a list of result dicts with keys:
      - model_name: the model name from the TOML
      - config_path: path to the TOML file
      - passed: number of functions that passed
      - hallucinated: total hallucinated lines
      - bonus: total bonus matched lines
      - runtime: total runtime in seconds
      - error: error message if the benchmark failed
    """
    results: list[dict] = []

    # Determine if corpus is a file path or corpus config name
    corpus_path = Path(corpus)
    if corpus_path.is_file():
        # It's a file path, use --file
        corpus_flag = "--file"
        corpus_value = str(corpus_path)
    else:
        # It's a corpus config name, use --corpus
        corpus_flag = "--corpus"
        corpus_value = corpus

    for i, config_path in enumerate(config_paths):
        content = config_path.read_text()
        model_name = read_toml_field(content, "name") or config_path.stem

        print(f"\n[{i + 1}/{len(config_paths)}] Benchmarking {model_name}...")

        if dry_run:
            print(
                f"  [DRY-RUN] Would run: pixi run python bench.py run {corpus_flag} {corpus_value} --model {config_path}"
            )
            results.append(
                {
                    "model_name": model_name,
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": 0.0,
                    "error": "dry-run (skipped)",
                }
            )
            if i < len(config_paths) - 1:
                print("  Sleeping 30 seconds...")
                time.sleep(30)
            continue

        start_time = time.monotonic()
        try:
            cmd = [
                "pixi",
                "run",
                "python",
                "bench.py",
                "run",
                corpus_flag,
                corpus_value,
                "--model",
                str(config_path),
            ]
            print(f"  Running: {' '.join(cmd)}")
            result = run_subprocess(cmd, check=False)
            stdout = result.stdout
            stderr = result.stderr
            end_time = time.monotonic()
            runtime = end_time - start_time

            if result.returncode != 0:
                print(
                    f"  ERROR: benchmark failed (exit code {result.returncode})"
                )
                if stderr:
                    print(f"  stderr: {stderr[:500]}")
                results.append(
                    {
                        "model_name": model_name,
                        "config_path": config_path,
                        "passed": 0,
                        "hallucinated": 0,
                        "bonus": 0,
                        "runtime": runtime,
                        "error": f"exit code {result.returncode}",
                    }
                )
            else:
                print(f"  Completed successfully ({runtime:.1f}s).")
                results.append(
                    {
                        "model_name": model_name,
                        "config_path": config_path,
                        "passed": 0,
                        "hallucinated": 0,
                        "bonus": 0,
                        "runtime": runtime,
                        "error": None,
                    }
                )

        except FileNotFoundError:
            end_time = time.monotonic()
            runtime = end_time - start_time
            print(
                "  ERROR: 'pixi' not found. Is pixi installed?", file=sys.stderr
            )
            results.append(
                {
                    "model_name": model_name,
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": runtime,
                    "error": "pixi not found",
                }
            )

        # Sleep 30 seconds between benchmarks (not after the last one)
        if i < len(config_paths) - 1:
            print("  Sleeping 30 seconds...")
            time.sleep(30)

    return results


def parse_results_from_files(
    corpus: str,
    results_dir: Path = Path("results"),
) -> list[dict]:
    """Parse benchmark results from generated JSON dump files.

    Returns a list of result dicts with model names and aggregated stats.
    """
    parsed_results: list[dict] = []

    if not results_dir.is_dir():
        return parsed_results

    # Find result files matching the corpus pattern
    corpus_stem = Path(corpus).stem if Path(corpus).is_file() else corpus

    for json_file in sorted(results_dir.glob(f"{corpus_stem}__*.json")):
        try:
            import json

            data = json.loads(json_file.read_text())
            results = data.get("results", [])

            if not results:
                continue

            passed = sum(1 for r in results if r.get("passed"))
            hallucinated = sum(r.get("hallucinated", 0) for r in results)
            bonus = sum(r.get("bonus_matched", 0) for r in results)

            # Extract model name from the file stem (everything before __)
            model_name = (
                json_file.stem.split("__")[0]
                if "__" in json_file.stem
                else json_file.stem
            )

            parsed_results.append(
                {
                    "model_name": model_name,
                    "passed": passed,
                    "hallucinated": hallucinated,
                    "bonus": bonus,
                    "error": None,
                }
            )

        except (json.JSONDecodeError, KeyError) as e:
            print(
                f"  (warning: could not parse {json_file.name}: {e})",
                file=sys.stderr,
            )

    return parsed_results


def generate_markdown_table(results: list[dict], save_path: Path) -> None:
    """Generate a Markdown table with pass, hallucinations, bonus, and runtime per model."""
    # Sort by pass count descending, then hallucinations ascending
    sorted_results = sorted(
        results,
        key=lambda r: (-r.get("passed", 0), r.get("hallucinated", 0)),
    )

    lines = [
        "| Model | Pass | Hallucinations | Bonus | Runtime (s) |",
        "|---|---|---|---|---|",
    ]

    for r in sorted_results:
        model = r.get("model_name", "unknown")
        passed = r.get("passed", 0)
        hallucinated = r.get("hallucinated", 0)
        bonus = r.get("bonus", 0)
        runtime = r.get("runtime", 0.0)
        error = r.get("error")

        if error:
            model = f"{model} (ERROR: {error})"

        lines.append(
            f"| {model} | {passed} | {hallucinated} | {bonus} | {runtime:.1f} |"
        )

    table_content = "\n".join(lines) + "\n"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table_content)
    print(f"\nResults table written to: {save_path}")


def run_visualization() -> None:
    """Run the visualization script to generate HTML dashboards."""
    try:
        result = run_subprocess(
            ["pixi", "run", "python", "analysis/visualize.py"]
        )
        if result.returncode == 0:
            print(
                "\nVisualization complete. Check analysis/charts/ for HTML dashboards."
            )
        else:
            print(
                f"\nVisualization failed (exit code {result.returncode}): {result.stderr}",
                file=sys.stderr,
            )
    except FileNotFoundError:
        print(
            "ERROR: 'pixi' not found. Cannot run visualization.",
            file=sys.stderr,
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  Benchmark Suite — LMStudio Model Orchestrator")
    print("=" * 60)
    print(f"  Corpus:        {args.corpus}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Corpora dir:   {args.corpora_dir}")
    print(f"  User models:   {args.user_models_dir}")
    print(f"  Template dir:  {args.template_dir}")
    print(f"  Dry run:       {args.dry_run}")
    print(f"  Save table:    {args.save_table}")
    print("=" * 60)

    # Step 1: Discover models
    print("\n[1/5] Discovering LMStudio models...")
    models = discover_models()
    print(f"  Found {len(models)} model(s):")
    for m in models:
        print(f"    - {m}")

    # Step 2: Generate user config TOMLs
    print("\n[2/5] Generating user config TOMLs...")
    config_paths = generate_user_configs(
        models=models,
        template_dir=args.template_dir,
        user_models_dir=args.user_models_dir,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
    )

    if not config_paths:
        print("No configs generated. Exiting.", file=sys.stderr)
        return 1

    # Step 3: Run benchmarks
    print("\n[3/5] Running benchmarks...")
    benchmark_results = run_benchmarks(
        config_paths=config_paths,
        corpus=args.corpus,
        dry_run=args.dry_run,
    )

    # Step 4: Parse results from JSON dump files (if any were generated)
    print("\n[4/5] Parsing benchmark results...")
    parsed_results = parse_results_from_files(args.corpus, Path("results"))

    # Merge parsed results with benchmark results (by model name)
    # For any model that wasn't parsed from a JSON file, use the benchmark result
    parsed_by_name = {r["model_name"]: r for r in parsed_results}
    for br in benchmark_results:
        name = br["model_name"]
        if name in parsed_by_name:
            # Use parsed values (more accurate)
            br["passed"] = parsed_by_name[name]["passed"]
            br["hallucinated"] = parsed_by_name[name]["hallucinated"]
            br["bonus"] = parsed_by_name[name]["bonus"]
            br["runtime"] = br.get("runtime", 0.0)  # Keep runtime from benchmark_results
        else:
            # No JSON file found, keep benchmark result (all zeros)
            pass

    # Step 5: Generate Markdown table
    print("\n[5/5] Generating Markdown results table...")
    generate_markdown_table(benchmark_results, args.save_table)

    # Step 6: Run visualization
    if not args.dry_run:
        print("\nGenerating visual report...")
        run_visualization()

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
