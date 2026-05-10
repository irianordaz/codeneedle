#!/usr/bin/env python3
"""Benchmark suite orchestrator for LMStudio models.

Discovers all models via `lms ls`, creates per-model config TOMLs from
templates in `configs/models/`, runs benchmarks sequentially with configurable
pauses, produces a Markdown results table (with runtime and totals), and
generates visual reports.

Usage:
    pixi run python benchmark_suite.py --corpus http_server
    pixi run python benchmark_suite.py --corpus jquery --max-tokens 128000
    pixi run python benchmark_suite.py --corpus jquery --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 32768
DEFAULT_CORPORA_DIR = Path("configs/corpora")
DEFAULT_USER_MODELS_DIR = Path("configs/models/user")
DEFAULT_TEMPLATE_DIR = Path("configs/models")
DEFAULT_SAVE_TABLE = Path("results_table.md")
DEFAULT_COOLDOWN_SECONDS = 30
DEFAULT_SUBPROCESS_TIMEOUT = 3600  # 1 hour per model, generous


class BenchmarkError(Exception):
    """Raised when a required tool is missing or a pre-condition fails."""


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
    """Format max_tokens as 65K, 128K, etc (for filenames only)."""
    rounded = round_to_nearest_thousands(value)
    return f"{rounded // 1000}K"


def sanitize_model_name(name: str) -> str:
    """Replace slashes and special chars with underscores for filenames."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def determine_framework(model_name: str) -> str:
    """Determine framework (gguf or mlx) from the model name."""
    if "mlx" in model_name.lower():
        return "mlx"
    return "gguf"


def _longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings (rolling-space)."""
    if not s1 or not s2:
        return ""

    m, n = len(s1), len(s2)
    max_len = 0
    end_idx = 0  # ending index in s1

    # Rolling rows: only need previous row
    prev_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
                if curr_row[j] > max_len:
                    max_len = curr_row[j]
                    end_idx = i
        prev_row = curr_row

    return s1[end_idx - max_len : end_idx]


def find_best_template(model_name: str, template_dir: Path) -> Path | None:
    """Find the best template TOML for a given LMStudio model name.

    Strategy (in priority order):
    1. Exact match (after normalization).
    2. Prefix/containment match (model stem in template stem or vice versa).
    3. Longest common substring (least preferred).

    Among ties at the same priority level, prefer templates whose framework
    (gguf vs mlx) matches the model's detected framework.
    """
    if not template_dir.is_dir():
        return None

    norm_core = normalize_name(model_name)

    best_match: Path | None = None
    best_score = (
        -1,
        False,
        0,
        "",
    )  # (is_exact, is_prefix, lcs_len, framework_match)

    for toml_file in template_dir.glob("*.toml"):
        stem = toml_file.stem
        norm_stem = normalize_name(stem)
        template_framework = determine_framework(stem)

        # Priority 1: Exact match (after normalization, ignoring case)
        is_exact = norm_core == norm_stem

        # Priority 2: Prefix/containment match
        is_prefix = (
            norm_core.startswith(norm_stem)
            or norm_stem.startswith(norm_core)
            or norm_core in norm_stem
            or norm_stem in norm_core
        )

        # Priority 3: Longest common substring
        common = _longest_common_substring(norm_core, norm_stem)
        lcs_len = len(common)

        # Framework match (used for tie-breaking within same priority)
        model_framework = determine_framework(model_name)
        framework_match = "1" if template_framework == model_framework else "0"

        # Build ranking tuple: (is_exact, is_prefix, lcs_len, framework_match)
        score = (is_exact, is_prefix, lcs_len, framework_match)

        if score > best_score:
            best_score = score
            best_match = toml_file

    return best_match


def update_max_tokens_in_toml(content: str, new_max_tokens: int) -> str:
    """Update the max_tokens value in a TOML file content string.

    Inserts a new line if the template lacks a max_tokens field.
    """
    if re.search(r"^max_tokens\s*=", content, re.MULTILINE):
        return re.sub(
            r"(max_tokens\s*=\s*)\d+",
            rf"\g<1>{new_max_tokens}",
            content,
        )
    # Insert max_tokens line after the name field (or at the top if no name)
    name_match = re.search(r"^name\s*=", content, re.MULTILINE)
    if name_match:
        insert_pos = name_match.end()
        return (
            content[:insert_pos]
            + f"\nmax_tokens = {new_max_tokens}"
            + content[insert_pos:]
        )
    return content + f"\nmax_tokens = {new_max_tokens}\n"


def create_minimal_toml(
    model_name: str, framework: str, max_tokens: int
) -> str:
    """Generate a minimal TOML config when no template is found.

    Uses the numeric integer value for max_tokens (not formatted as 65K)
    so tomllib can parse it.
    """
    return (
        f'name = "{model_name}"\n'
        f'base_url = "http://localhost:1234"\n'
        f"temperature = 0.0\n"
        f"max_tokens = {max_tokens}\n"
        f"timeout = 10000.0\n"
        f"suppress_thinking = true\n"
    )


def update_toml_name_field(content: str, new_name: str) -> str:
    """Update the `name` field in a TOML file content string."""
    return re.sub(
        r'^(name\s*=\s*)["\'][^"\']+["\']',
        rf"\g<1>{json.dumps(new_name)}",
        content,
        count=1,
        flags=re.MULTILINE,
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
    """Parse the output of `lms ls` to extract all installed model names.

    Handles LMStudio's tabular output format. Returns ALL installed models
    (both loaded and unloaded). Embedding models are excluded.
    """
    models = []
    in_llm_section = False

    for line in output.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip the summary line ("You have N models, taking X GiB")
        if "models, taking" in stripped:
            continue

        # Detect section headers (LLM / EMBEDDING) — may have trailing content
        if re.match(r"^LLM\s*$", stripped, re.IGNORECASE):
            in_llm_section = True
            continue
        if re.match(r"^EMBEDDING\s*$", stripped, re.IGNORECASE):
            in_llm_section = False
            continue

        # Skip header lines (LLM/EMBEDDING PARAMS...)
        if re.match(r"^(LLM|EMBEDDING)\s+\w+\s+", stripped, re.IGNORECASE):
            # If we see "LLM PARAMS..." set flag, if "EMBEDDING PARAMS..." clear it
            if re.match(r"^LLM\s+", stripped, re.IGNORECASE):
                in_llm_section = True
            else:
                in_llm_section = False
            continue

        # Only include models in the LLM section
        if not in_llm_section:
            continue

        # Extract model name from the first column
        # Format: "model/name (1 variant)" or "model/name"
        match = re.match(r"^(\S+?)(?:\s*\([^)]*\))?\s+", stripped)
        if match:
            models.append(match.group(1))

    return models


def parse_loaded_models(output: str) -> list[str]:
    """Parse the output of `lms ls` to extract only LOADED model names."""
    loaded = []
    in_llm_section = False

    for line in output.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if "models, taking" in stripped:
            continue

        if re.match(r"^LLM\s*$", stripped, re.IGNORECASE):
            in_llm_section = True
            continue
        if re.match(r"^EMBEDDING\s*$", stripped, re.IGNORECASE):
            in_llm_section = False
            continue

        if re.match(r"^(LLM|EMBEDDING)\s+\w+\s+", stripped, re.IGNORECASE):
            if re.match(r"^LLM\s+", stripped, re.IGNORECASE):
                in_llm_section = True
            else:
                in_llm_section = False
            continue

        if not in_llm_section:
            continue

        # Only include models that are LOADED
        if "✓ LOADED" not in stripped and "LOADED" not in stripped:
            continue

        match = re.match(r"^(\S+?)(?:\s*\([^)]*\))?\s+", stripped)
        if match:
            loaded.append(match.group(1))

    return loaded


def validate_corpus(corpus: str, corpora_dir: Path) -> None:
    """Validate that the corpus argument can be resolved.

    Raises BenchmarkError if the corpus is not found.
    """
    corpus_path = Path(corpus)
    if corpus_path.is_file():
        return  # File path is valid
    # Check if it's a corpus config name
    candidate = corpora_dir / f"{corpus}.toml"
    if candidate.is_file():
        return
    # List available corpora for a helpful error
    available = (
        [p.stem for p in corpora_dir.glob("*.toml")]
        if corpora_dir.is_dir()
        else []
    )
    raise BenchmarkError(
        f"Corpus '{corpus}' not found in {corpora_dir}.\n"
        f"Available corpora: {', '.join(available) if available else '(none)'}"
    )


def run_subprocess(
    cmd: list[str],
    check: bool = False,
    timeout: int | None = None,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return the result.

    When `stream=True`, stdout/stderr are written live to the console.
    """
    kwargs: dict = {
        "capture_output": not stream,
        "text": True,
        "check": check,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    if stream:
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
        )
        proc.wait()
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout="",
            stderr="",
        )
    else:
        return subprocess.run(cmd, **kwargs)


def load_model(model_name: str, context_length: int = 131072) -> bool:
    """Load a model in LM Studio. Returns True on success."""
    cmd = [
        "lms",
        "load",
        model_name,
        "--context-length",
        str(context_length),
        "--gpu",
        "max",
        "--ttl",
        "99999",
        "-y",
    ]
    try:
        result = run_subprocess(cmd, check=False, stream=True)
        if result.returncode == 0:
            print(f"  Loaded: {model_name}")
            return True
        else:
            print(
                f"  ERROR loading {model_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}", file=sys.stderr)
        return False


def unload_model(model_name: str) -> bool:
    """Unload a model from LM Studio. Returns True on success."""
    cmd = ["lms", "unload", model_name]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input="y\n",
        )
        if result.returncode == 0:
            print(f"  Unloaded: {model_name}")
            return True
        else:
            # Model may not be loaded — that's OK
            if "not found" in (result.stderr or "").lower():
                return True
            print(
                f"  WARNING unloading {model_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False
    except Exception as e:
        print(f"  WARNING unloading {model_name}: {e}", file=sys.stderr)
        return False


def unload_all_loaded() -> list[str]:
    """Unload all currently loaded models. Returns list of unloaded model names."""
    try:
        result = run_subprocess(["lms", "ls"])
        if result.returncode != 0:
            print(
                f"WARNING: 'lms ls' failed ({result.returncode}), "
                "cannot determine loaded models.",
                file=sys.stderr,
            )
            return []
    except FileNotFoundError:
        print(
            "WARNING: 'lms' not found, skipping preload cleanup.",
            file=sys.stderr,
        )
        return []

    loaded = parse_loaded_models(result.stdout)
    if not loaded:
        return []

    unloaded = []
    for model in loaded:
        if unload_model(model):
            unloaded.append(model)
    return unloaded


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
        help="Corpus config name (e.g. 'http_server') or path to a Python source file.",
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
        "--clean-run",
        action="store_true",
        help="Delete existing results tables before generating new ones.",
    )
    parser.add_argument(
        "--save-table",
        type=Path,
        default=DEFAULT_SAVE_TABLE,
        help=f"Path to write the Markdown results table (default: {DEFAULT_SAVE_TABLE}).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Seconds to wait between benchmarks (default: {DEFAULT_COOLDOWN_SECONDS}).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only benchmark models whose name contains this string (case-insensitive).",
    )
    parser.add_argument(
        "--includes",
        type=str,
        default=None,
        help="Comma-delimited list of keywords; only models containing ANY of these keywords will be benchmarked (case-insensitive).",
    )
    parser.add_argument(
        "--excludes",
        type=str,
        default=None,
        help="Comma-delimited list of keywords; models containing ANY of these keywords will be excluded (case-insensitive).",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=131072,
        help="Context length to use when loading models (default: 131072).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def discover_models() -> list[str]:
    """Discover all installed models via `lms ls`."""
    try:
        result = run_subprocess(["lms", "ls"])
    except FileNotFoundError:
        raise BenchmarkError(
            "'lms' executable not found. Is LMStudio installed and on PATH?"
        )

    if result.returncode != 0:
        raise BenchmarkError(
            f"'lms ls' failed with code {result.returncode}: {result.stderr}"
        )

    models = parse_lms_ls(result.stdout)
    if not models:
        raise BenchmarkError(
            "No models found via 'lms ls'. Install a model in LM Studio first."
        )

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
    seen_filenames: dict[str, Path] = {}  # For collision detection

    for model_name in models:
        framework = determine_framework(model_name)
        formatted_max_tokens = format_max_tokens(max_tokens)
        safe_name = sanitize_model_name(model_name)

        toml_filename = f"{safe_name}-{framework}-{formatted_max_tokens}.toml"

        # Collision detection: sanitize_model_name can collapse different models to the same filename (e.g. "qwen/qwen3.6" and "qwen-qwen3.6"). Append a short hash if collision detected.
        if toml_filename in seen_filenames:
            import hashlib

            suffix = hashlib.md5(model_name.encode()).hexdigest()[:6]
            base, ext = toml_filename.rsplit(".", 1)
            toml_filename = f"{base}_{suffix}.{ext}"

        toml_path = user_models_dir / toml_filename
        seen_filenames[toml_filename] = toml_path

        if dry_run:
            print(f"  [DRY-RUN] Would create: {toml_path}")
            created.append(toml_path)
            continue

        # Find best template
        template = find_best_template(model_name, template_dir)

        if template is not None:
            content = template.read_text()
            content = update_max_tokens_in_toml(content, max_tokens)
            # Bug #3 fix: update the name field to the discovered LMStudio model id
            content = update_toml_name_field(content, model_name)
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
    cooldown_seconds: int,
    dry_run: bool = False,
    subprocess_timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
    context_length: int = 131072,
) -> list[dict]:
    """Run benchmarks for each config, sleeping between runs.

    Loads each model before benchmarking and unloads it after.
    Returns a list of result dicts with keys:
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
        if config_path.is_file():
            model_name = (
                read_toml_field(config_path.read_text(), "name")
                or config_path.stem
            )
        else:
            model_name = config_path.stem

        print(f"\n[{i + 1}/{len(config_paths)}] Benchmarking {model_name}...")

        if dry_run:
            print(
                f"  [DRY-RUN] Would run: pixi run python bench.py run {corpus_flag} {corpus_value} --model {config_path}"
            )
            results.append(
                {
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": 0.0,
                    "error": None,
                }
            )
            continue

        # Load the model before benchmarking
        if not load_model(model_name, context_length):
            print(
                f"  SKIPPED: failed to load {model_name}.",
                file=sys.stderr,
            )
            results.append(
                {
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": 0.0,
                    "error": "model load failed",
                }
            )
            # Sleep between benchmarks (not after the last one)
            if i < len(config_paths) - 1:
                print(f"  Sleeping {cooldown_seconds} seconds...")
                time.sleep(cooldown_seconds)
            continue

        start_time = time.monotonic()
        benchmark_ok = False
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
            # Stream output live so the user sees progress
            result = run_subprocess(cmd, check=False, stream=True)
            end_time = time.monotonic()
            runtime = end_time - start_time
            benchmark_ok = True

            # Bug #1 fix: bench.py returns exit code 1 when not all functions
            # passed, NOT when the benchmark crashed. Treat exit code 1 as
            # "completed with some failures" rather than "error". Only exit
            # codes other than 0 and 1 indicate a real failure (e.g. server
            # unreachable, config parse error, etc.).
            if result.returncode == 0:
                print(f"  Completed successfully ({runtime:.1f}s).")
            elif result.returncode == 1:
                print(
                    f"  Completed with some failures ({runtime:.1f}s). "
                    f"(exit code 1 = not all functions passed, see JSON dump)"
                )
            else:
                print(
                    f"  ERROR: benchmark crashed (exit code {result.returncode})"
                )

            # Always record the result — even exit code 1 means the run
            # completed, we just need to parse the JSON dump for stats.
            results.append(
                {
                    "config_path": config_path,
                    "passed": 0,  # Will be filled by merge step
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": runtime,
                    "error": None
                    if result.returncode <= 1
                    else f"exit code {result.returncode}",
                }
            )

        except subprocess.TimeoutExpired:
            end_time = time.monotonic()
            runtime = end_time - start_time
            print(
                f"  ERROR: benchmark timed out after {subprocess_timeout}s.",
                file=sys.stderr,
            )
            results.append(
                {
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": runtime,
                    "error": f"timed out after {subprocess_timeout}s",
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
                    "config_path": config_path,
                    "passed": 0,
                    "hallucinated": 0,
                    "bonus": 0,
                    "runtime": runtime,
                    "error": "pixi not found",
                }
            )
        finally:
            # Always unload the model after benchmarking
            unload_model(model_name)

        # Sleep between benchmarks (not after the last one)
        if i < len(config_paths) - 1:
            print(f"  Sleeping {cooldown_seconds} seconds...")
            time.sleep(cooldown_seconds)

    return results


def parse_results_from_files(
    corpus: str,
    user_models_dir: Path,
    results_dir: Path = Path("results"),
) -> list[dict]:
    """Parse benchmark results from generated JSON dump files.

    Keys by the config path in user_models_dir so it matches what
    `run_benchmarks` uses for merging.
    """
    parsed_results: list[dict] = []

    if not results_dir.is_dir():
        return parsed_results

    # Find result files matching the corpus pattern
    corpus_stem = Path(corpus).stem if Path(corpus).is_file() else corpus

    for json_file in sorted(results_dir.glob(f"{corpus_stem}__*.json")):
        data = json.loads(json_file.read_text())
        results = data.get("results", [])

        if not results:
            continue

        passed = sum(1 for r in results if r.get("passed"))
        hallucinated = sum(r.get("hallucinated", 0) for r in results)
        bonus = sum(r.get("bonus_matched", 0) for r in results)
        primary_matched = sum(r.get("primary_matched", 0) for r in results)
        runtime = sum(r.get("latency_s", 0.0) for r in results)

        # JSON dump files are named {corpus}__{model-stem}.json.
        # Extract the model-stem (part after '__') and reconstruct
        # the config path in user_models_dir.
        model_stem = (
            json_file.stem.split("__")[1]
            if "__" in json_file.stem
            else json_file.stem
        )

        parsed_results.append(
            {
                "config_path": user_models_dir / f"{model_stem}.toml",
                "passed": passed,
                "hallucinated": hallucinated,
                "bonus": bonus,
                "primary_matched": primary_matched,
                "runtime": runtime,
                "error": None,
            }
        )

    return parsed_results


def merge_results(
    existing: list[dict],
    new: list[dict],
) -> list[dict]:
    """Merge new benchmark results with existing results.

    Matches results by `config_path` key so that results from
    different runs (different models) can be combined into a
    single table.

    Existing (parsed) results take priority over new (placeholder)
    results because the new results from run_benchmarks have
    passed/hallucinated/bonus = 0 as placeholders that need to
    be filled in by the parsed JSON dump data.

    Args:
        existing: Previously parsed results (from parse_results_from_files).
        new: Newly benchmarked results (from run_benchmarks).

    Returns:
        Combined list of results, with existing data overwriting
        entries that share the same config_path.
    """
    # Index new results by config_path for O(1) lookup
    new_map: dict[str, dict] = {}
    for r in new:
        cp = str(r.get("config_path", ""))
        new_map[cp] = r

    # Overwrite with existing (parsed) data — existing takes priority
    for r in existing:
        cp = str(r.get("config_path", ""))
        new_map[cp] = r

    return list(new_map.values())


def _build_table_data(results: list[dict]) -> dict:
    """Build shared table data from benchmark results."""
    sorted_results = sorted(
        results,
        key=lambda r: (-r.get("passed", 0), r.get("hallucinated", 0)),
    )

    header = ["Model", "Pass", "Hallucinations", "Bonus", "Primary", "Runtime (s)"]
    rows: list[list[str]] = []

    total_passed = 0
    total_hallucinated = 0
    total_bonus = 0
    total_primary = 0
    total_runtime = 0.0

    for r in sorted_results:
        config_path = r.get("config_path", Path("unknown"))
        model = config_path.stem
        passed = r.get("passed", 0)
        hallucinated = r.get("hallucinated", 0)
        bonus = r.get("bonus", 0)
        primary = r.get("primary_matched", 0)
        runtime = r.get("runtime", 0.0)
        error = r.get("error")

        if error:
            model = f"{model} (ERROR: {error})"

        rows.append([model, str(passed), str(hallucinated), str(bonus), str(primary), f"{runtime:.1f}"])

        total_passed += passed
        total_hallucinated += hallucinated
        total_bonus += bonus
        total_primary += primary
        total_runtime += runtime

    totals = ["Total", str(total_passed), str(total_hallucinated), str(total_bonus), str(total_primary), f"{total_runtime:.1f}"]
    return {
        "header": header,
        "rows": rows,
        "totals": totals,
        "corpus": "",
    }


def generate_markdown_table(
    results: list[dict],
    save_path: Path,
    corpus: str = "",
) -> None:
    """Generate a Markdown table with improved formatting.

    Appends new rows between markers so previous runs are preserved.
    Uses equal-width columns and one row per model for readability.
    """
    data = _build_table_data(results)
    data["corpus"] = corpus

    header = data["header"]
    rows = data["rows"]
    totals = data["totals"]

    # Compute max width per column
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    for i, cell in enumerate(totals):
        col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(labels: list[str], bold: bool = False) -> str:
        parts = []
        for i, label in enumerate(labels):
            pad = col_widths[i] - len(label)
            if bold:
                parts.append((" **" + label + "**" + " " * pad))
            else:
                parts.append((" " + label + " " + " " * pad))
        return "|" + "|".join(parts) + "|"

    def _fmt_separator() -> str:
        parts = []
        for i in range(len(header)):
            pad = col_widths[i] - len(header[i])
            parts.append(("-" * (2 + pad)))
        return "|" + "|".join(parts) + "|"

    new_block_lines = [
        f"<!-- Corpus: {corpus} -->" if corpus else "",
        "",
        _fmt_row(header),
        _fmt_separator(),
    ]

    for row in rows:
        new_block_lines.append(_fmt_row(row))

    new_block_lines.append(_fmt_row(totals, bold=True))
    new_block_lines.append("")

    new_block = "\n".join(new_block_lines)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        existing = save_path.read_text()
        marker_start = "<!-- BENCHMARK_ROWS_START -->"
        marker_end = "<!-- BENCHMARK_ROWS_END -->"
        start_idx = existing.find(marker_start)
        end_idx = existing.find(marker_end)

        if start_idx != -1 and end_idx != -1:
            before = existing[: start_idx + len(marker_start)]
            after = existing[end_idx:]
            table_content = before + new_block + after
        else:
            # No markers yet — build complete table with markers
            header_line = _fmt_row(header)
            separator_line = _fmt_separator()
            table_content = (
                f"<!-- Corpus: {corpus} -->"
                + "\n\n"
                + header_line
                + "\n"
                + separator_line
                + "\n"
                + "<!-- BENCHMARK_ROWS_START -->"
                + "\n"
                + new_block
                + "\n"
                + "<!-- BENCHMARK_ROWS_END -->"
            )
    else:
        header_line = _fmt_row(header)
        separator_line = _fmt_separator()
        table_content = (
            f"<!-- Corpus: {corpus} -->"
            + "\n\n"
            + header_line
            + "\n"
            + separator_line
            + "\n"
            + "<!-- BENCHMARK_ROWS_START -->"
            + "\n"
            + new_block
            + "\n"
            + "<!-- BENCHMARK_ROWS_END -->"
        )

    save_path.write_text(table_content)
    print(f"\nResults table written to: {save_path}")


def _build_full_html(
    header: list[str],
    header_cells: list[str],
    html_rows: list[str],
    totals: list[str],
    col_widths_px: list[int],
    corpus: str,
) -> str:
    """Build a complete HTML file from scratch."""
    total_cells = []
    for i, cell in enumerate(totals):
        width = col_widths_px[i]
        total_cells.append(
            f'<td style="width: {width}px; padding: 6px 12px; text-align: left; '
            f'font-weight: bold; border-top: 2px solid #333;">**{cell}**</td>'
        )
    html_total = "<tr>" + "".join(total_cells) + "</tr>"

    title = f"Results Table{f' — {corpus}' if corpus else ''}"
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results Table</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            padding: 24px;
            margin: 0;
        }}
        .table-container {{
            max-width: 960px;
            margin: 0 auto;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        h2 {{
            margin: 0;
            padding: 16px 24px;
            background: #1a73e8;
            color: #fff;
            font-size: 18px;
            font-weight: 500;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        thead th {{
            background: #f8f9fa;
            padding: 10px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            color: #555;
            border-bottom: 2px solid #e0e0e0;
        }}
        tbody td {{
            padding: 10px 12px;
            font-size: 14px;
            border-bottom: 1px solid #eee;
        }}
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        tbody tr:last-child td {{
            border-bottom: none;
        }}
        tfoot td {{
            padding: 12px;
            font-weight: 600;
            background: #f8f9fa;
            border-top: 2px solid #1a73e8;
        }}
        thead th {{
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        thead th:hover {{
            background: #e8f4fd;
        }}
        thead th .sort-arrow {{
            display: inline-block;
            margin-left: 6px;
            font-size: 10px;
            opacity: 0.4;
            transition: opacity 0.2s;
        }}
        thead th.active-sort .sort-arrow {{
            opacity: 1;
            color: #1a73e8;
        }}
    </style>
</head>
<body>
    <div class="table-container">
        <h2>{title}</h2>
        <table>
            <thead>
                <tr>
{"".join(header_cells)}
                </tr>
            </thead>
            <tbody id="benchmark-rows">
{"".join(html_rows)}
{html_total}
            </tbody><!-- BENCHMARK_ROWS_END -->
        </table>
    </div>
    <script>
    (function() {{
        let sortCol = -1;
        let sortAsc = true;

        function parseNumeric(val) {{
            const cleaned = val.replace(/,/g, '').replace(/\\(.*\\)/, '').trim();
            const num = parseFloat(cleaned);
            return isNaN(num) ? null : num;
        }}

        function sortTable(colIndex) {{
            const tbody = document.getElementById("benchmark-rows");
            if (!tbody) return;

            const rows = Array.from(tbody.querySelectorAll("tr"));
            const totalRow = rows.pop();

            if (colIndex === sortCol) {{
                sortAsc = !sortAsc;
            }} else {{
                sortCol = colIndex;
                sortAsc = true;
            }}

            rows.sort((a, b) => {{
                const cellsA = a.querySelectorAll("td");
                const cellsB = b.querySelectorAll("td");
                if (colIndex >= cellsA.length || colIndex >= cellsB.length) return 0;

                const valA = cellsA[colIndex].textContent.trim();
                const valB = cellsB[colIndex].textContent.trim();

                const numA = parseNumeric(valA);
                const numB = parseNumeric(valB);

                if (numA !== null && numB !== null) {{
                    return sortAsc ? numA - numB : numB - numA;
                }}

                const comparison = valA.localeCompare(valB);
                return sortAsc ? comparison : -comparison;
            }});

            tbody.innerHTML = rows.map(r => r.outerHTML).join("\\n");
            tbody.appendChild(totalRow);

            document.querySelectorAll("thead th").forEach((th, i) => {{
                th.classList.remove("active-sort");
                const arrow = th.querySelector(".sort-arrow");
                if (arrow) arrow.textContent = " \\u25B2\\u25BC";
            }});

            const activeTh = document.querySelectorAll("thead th")[colIndex];
            if (activeTh) {{
                activeTh.classList.add("active-sort");
                const arrow = activeTh.querySelector(".sort-arrow");
                if (arrow) arrow.textContent = sortAsc ? "\\u25B2" : "\\u25BC";
            }}
        }}

        document.querySelectorAll("thead th").forEach((th, i) => {{
            th.addEventListener("click", () => sortTable(i));
            const arrow = document.createElement("span");
            arrow.className = "sort-arrow";
            arrow.textContent = " \\u25B2\\u25BC";
            th.appendChild(arrow);
        }});
    }})();
    </script>
</body>
</html>"""
    return html_content


def generate_html_table(
    results: list[dict],
    save_path: Path,
    corpus: str = "",
) -> None:
    """Generate an HTML table with the same data as the Markdown table.

    Appends new rows between markers so previous runs are preserved.
    """
    data = _build_table_data(results)
    data["corpus"] = corpus

    header = data["header"]
    rows = data["rows"]
    totals = data["totals"]

    # Compute column widths for consistent styling
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    for i, cell in enumerate(totals):
        col_widths[i] = max(col_widths[i], len(cell))

    # Compute column widths in pixels (rough estimate: 8px per char + padding)
    col_widths_px = [max(w * 8 + 20, 80) for w in col_widths]

    html_rows = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            width = col_widths_px[i]
            cells.append(
                f'<td style="width: {width}px; padding: 6px 12px; text-align: left; '
                f'border-bottom: 1px solid #ddd; vertical-align: top;">{cell}</td>'
            )
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    total_cells = []
    for i, cell in enumerate(totals):
        width = col_widths_px[i]
        total_cells.append(
            f'<td style="width: {width}px; padding: 6px 12px; text-align: left; '
            f'font-weight: bold; border-top: 2px solid #333;">**{cell}**</td>'
        )
    html_total = "<tr>" + "".join(total_cells) + "</tr>"

    header_cells = []
    for i, h in enumerate(header):
        width = col_widths_px[i]
        header_cells.append(f'<th data-column="{i}" style="width: {width}px;">{h}</th>')

    new_tbody = "\n".join(html_rows)
    new_total = f"\n{html_total}\n"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        existing = save_path.read_text()
        marker_start = "<tbody id=\"benchmark-rows\">"
        marker_end = "</tbody><!-- BENCHMARK_ROWS_END -->"
        start_idx = existing.find(marker_start)
        end_idx = existing.find(marker_end)

        if start_idx != -1 and end_idx != -1:
            thead_start = existing.find("<thead>")
            thead_end = existing.find("</thead>")
            if thead_start != -1 and thead_end != -1:
                old_thead = existing[thead_start: thead_end + len("</thead>")]
                new_thead = "<thead>" + "".join(header_cells) + "</thead>"
                existing = existing.replace(old_thead, new_thead)

            # Remove any existing sorting script and body tag
            existing = existing.replace("</script>", "", 1)
            existing = existing.replace("</body>", "", 1)

            before = existing[: start_idx + len(marker_start)]
            after = existing[end_idx:]
            html_content = (
                before
                + new_tbody
                + new_total
                + after
            )
            # Append the sorting script at the end of the body
            html_content = html_content.rstrip() + "\n    </body>\n</html>"
            html_content = html_content.replace(
                "</html>",
                '''    <script>
    (function() {
        let sortCol = -1;
        let sortAsc = true;

        function parseNumeric(val) {
            const cleaned = val.replace(/,/g, '').replace(/\\(.*\\)/, '').trim();
            const num = parseFloat(cleaned);
            return isNaN(num) ? null : num;
        }

        function sortTable(colIndex) {
            const tbody = document.getElementById("benchmark-rows");
            if (!tbody) return;

            const rows = Array.from(tbody.querySelectorAll("tr"));
            const totalRow = rows.pop();

            if (colIndex === sortCol) {
                sortAsc = !sortAsc;
            } else {
                sortCol = colIndex;
                sortAsc = true;
            }

            rows.sort((a, b) => {
                const cellsA = a.querySelectorAll("td");
                const cellsB = b.querySelectorAll("td");
                if (colIndex >= cellsA.length || colIndex >= cellsB.length) return 0;

                const valA = cellsA[colIndex].textContent.trim();
                const valB = cellsB[colIndex].textContent.trim();

                const numA = parseNumeric(valA);
                const numB = parseNumeric(valB);

                if (numA !== null && numB !== null) {
                    return sortAsc ? numA - numB : numB - numA;
                }

                const comparison = valA.localeCompare(valB);
                return sortAsc ? comparison : -comparison;
            });

            tbody.innerHTML = rows.map(r => r.outerHTML).join("\\n");
            tbody.appendChild(totalRow);

            document.querySelectorAll("thead th").forEach((th, i) => {
                th.classList.remove("active-sort");
                const arrow = th.querySelector(".sort-arrow");
                if (arrow) arrow.textContent = " \\u25B2\\u25BC";
            });

            const activeTh = document.querySelectorAll("thead th")[colIndex];
            if (activeTh) {
                activeTh.classList.add("active-sort");
                const arrow = activeTh.querySelector(".sort-arrow");
                if (arrow) arrow.textContent = sortAsc ? "\\u25B2" : "\\u25BC";
            }
        }

        document.querySelectorAll("thead th").forEach((th, i) => {
            th.addEventListener("click", () => sortTable(i));
            const arrow = document.createElement("span");
            arrow.className = "sort-arrow";
            arrow.textContent = " \\u25B2\\u25BC";
            th.appendChild(arrow);
        });
    }})();
</script>
</html>''',
                1,
            )
        else:
            html_content = _build_full_html(
                header, header_cells, html_rows, totals, col_widths_px, corpus
            )
    else:
        html_content = _build_full_html(
            header, header_cells, html_rows, totals, col_widths_px, corpus
        )

    save_path.write_text(html_content)
    print(f"HTML results table written to: {save_path}")


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
    print(f"  Corpus:          {args.corpus}")
    print(f"  Max tokens:      {args.max_tokens}")
    print(f"  Corpora dir:     {args.corpora_dir}")
    print(f"  User models:     {args.user_models_dir}")
    print(f"  Template dir:    {args.template_dir}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Clean run:       {args.clean_run}")
    print(f"  Save table:      {args.save_table}")
    print(f"  Cooldown (s):    {args.cooldown_seconds}")
    print(f"  Filter:          {args.filter or '(all)'}")
    includes_list = [k.strip() for k in args.includes.split(",") if k.strip()] if args.includes else []
    excludes_list = [k.strip() for k in args.excludes.split(",") if k.strip()] if args.excludes else []
    print(f"  Includes:        {', '.join(includes_list) if includes_list else '(all)'}")
    print(f"  Excludes:        {', '.join(excludes_list) if excludes_list else '(none)'}")
    print(f"  Context length:  {args.context_length}")
    print("=" * 60)

    # Handle --clean-run: create initial marker files and delete corpus results
    if args.clean_run:
        print("\n[Clean run] Removing previous results tables...")
        save_table = args.save_table
        html_table_path = save_table.with_suffix(".html")
        if save_table.exists():
            save_table.unlink()
            print(f"  Removed: {save_table}")
        if html_table_path.exists():
            html_table_path.unlink()
            print(f"  Removed: {html_table_path}")

        results_dir = Path("results")
        if results_dir.exists():
            corpus_prefix = f"{args.corpus}__"
            removed = 0
            for f in results_dir.glob("*.json"):
                if f.name.startswith(corpus_prefix):
                    f.unlink()
                    removed += 1
                    print(f"  Removed: {f}")
            print(f"  Removed {removed} result file(s) for corpus '{args.corpus}'.")

    try:
        # Step 1: Discover models
        print("\n[1/7] Discovering LMStudio models...")
        models = discover_models()
        if args.filter:
            f = args.filter.lower()
            models = [m for m in models if f in m.lower()]
            if not models:
                print(
                    f"No models match filter '{args.filter}'.",
                    file=sys.stderr,
                )
                return 1

        # Apply --includes filter
        if includes_list:
            filtered = []
            for m in models:
                m_lower = m.lower()
                if any(kw in m_lower for kw in includes_list):
                    filtered.append(m)
            models = filtered
            if not models:
                print(
                    f"No models match includes {[', '.join(includes_list)]}.",
                    file=sys.stderr,
                )
                return 1

        # Apply --excludes filter
        if excludes_list:
            filtered = []
            for m in models:
                m_lower = m.lower()
                if not any(kw in m_lower for kw in excludes_list):
                    filtered.append(m)
            models = filtered
            if not models:
                print(
                    f"All models excluded by {[', '.join(excludes_list)]}.",
                    file=sys.stderr,
                )
                return 1
        print(f"  Found {len(models)} model(s):")
        for m in models:
            print(f"    - {m}")

        # Step 2: Validate corpus
        print(f"\n[2/7] Validating corpus '{args.corpus}'...")
        validate_corpus(args.corpus, args.corpora_dir)
        print("  Corpus resolved OK.")

        # Step 3: Generate user config TOMLs
        print("\n[3/7] Generating user config TOMLs...")
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

        # Step 3.5: Unload all loaded models so we always load one at a time
        print("\n[3.5/7] Unloading all loaded models...")
        unloaded = unload_all_loaded()
        if unloaded:
            print(f"  Unloaded {len(unloaded)} model(s) to ensure clean state.")
        else:
            print("  No models were loaded (clean state).")

        # Step 4: Run benchmarks (with per-model load/unload)
        print("\n[4/7] Running benchmarks...")
        benchmark_results = run_benchmarks(
            config_paths=config_paths,
            corpus=args.corpus,
            cooldown_seconds=args.cooldown_seconds,
            dry_run=args.dry_run,
            context_length=args.context_length,
        )

        # Step 5: Parse results from JSON dump files (if any were generated)
        print("\n[5/7] Parsing benchmark results...")
        parsed_results = parse_results_from_files(
            args.corpus, args.user_models_dir, Path("results")
        )

        # Merge parsed results with benchmark results (by config_path)
        benchmark_results = merge_results(parsed_results, benchmark_results)

        # Step 6: Generate Markdown table
        print("\n[6/7] Generating Markdown results table...")
        generate_markdown_table(benchmark_results, args.save_table, args.corpus)

        # Step 6b: Generate HTML table
        html_table_path = args.save_table.with_suffix(".html")
        print("\n[6b/7] Generating HTML results table...")
        generate_html_table(benchmark_results, html_table_path, args.corpus)

        # Step 7: Run visualization (if not dry-run)
        if not args.dry_run:
            print("\n[7/7] Generating visual report...")
            run_visualization()

    except BenchmarkError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
