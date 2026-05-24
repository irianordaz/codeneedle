#!/usr/bin/env python3
"""Benchmark suite orchestrator for LMStudio, Ollama, and llama.cpp models.

Discovers models from each runner, creates per-model config TOMLs from
templates in configs/models/, runs benchmarks sequentially with configurable
pauses, and produces Markdown + HTML results tables with per-model runtime,
token counts, and scores.

Runners
-------
lmstudio   Discovers models via `lms ls`. Loads/unloads via `lms load/unload`.
           Requires LM Studio with the local server running (port 1234).
ollama     Discovers models via `ollama ls`. Loads/unloads via `ollama pull/stop`.
           Requires Ollama running (port 11434).
llama.cpp  Discovers *.gguf files under ~/.lmstudio/models (skipping mmproj
           projector files). Spawns a llama-server process per model on port
           8080 using all available GPU layers (-ngl 999).

By default all three runners are used. Pass --runner to restrict to one or more.

Usage
-----
# Run all runners against the http_server corpus:
    pixi run python benchmark_suite.py --corpus http_server

# Run only llama.cpp models, skipping anything smaller than 25 B parameters:
    pixi run python benchmark_suite.py --corpus http_server --runner llama.cpp --min-size 25B

# Run only Qwen models across all runners (case-insensitive):
    pixi run python benchmark_suite.py --corpus http_server --includes qwen

# For llama.cpp, --includes/--excludes match against the full path under
# ~/.lmstudio/models, so directory names are searchable too:
    pixi run python benchmark_suite.py --corpus http_server --runner llama.cpp --includes MTP

# Filter the results table to show only models matching "qwen" or "gemma":
    pixi run python benchmark_suite.py --corpus http_server --show qwen,gemma

# Preview what would run without executing anything:
    pixi run python benchmark_suite.py --corpus http_server --dry-run

# Rebuild the results table from existing JSON dumps without re-running models:
    pixi run python benchmark_suite.py --corpus http_server --recreate-table

# Wipe previous results for this corpus, then run fresh:
    pixi run python benchmark_suite.py --corpus http_server --clean-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bench.tokens import count_tokens

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CORPORA_DIR = Path("configs/corpora")
DEFAULT_USER_MODELS_DIR = Path("configs/models/user")
DEFAULT_TEMPLATE_DIR = Path("configs/models")
DEFAULT_SAVE_TABLE = Path("results_table.md")
DEFAULT_COOLDOWN_SECONDS = 30
DEFAULT_SUBPROCESS_TIMEOUT = 10000  # ~2.8 hours per model
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLAMACPP_BASE_URL = "http://localhost:8080"
DEFAULT_LLAMACPP_PORT = 8080
DEFAULT_LLAMACPP_MODELS_DIR = Path.home() / ".lmstudio" / "models"
DEFAULT_LOAD_VERIFY_TIMEOUT = 180.0
LOAD_VERIFY_RETRY_INTERVAL = 3.0


class BenchmarkError(Exception):
    """Raised when a required tool is missing or a pre-condition fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Remove special characters for fuzzy matching (lowercase, no punctuation)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def sanitize_model_name(name: str) -> str:
    """Replace slashes and special chars with underscores for filenames."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def determine_framework(model_name: str) -> str:
    """Determine framework (gguf or mlx) from the model name."""
    if "mlx" in model_name.lower():
        return "mlx"
    return "gguf"


def parse_model_params(toml_path: Path) -> int | None:
    """Read number_of_parameters from a TOML config file.

    Returns the parameter count in billions, or None if not found.
    """
    if not toml_path.is_file():
        return None
    content = toml_path.read_text()
    match = re.search(
        r"^number_of_parameters\s*=\s*(\d+)", content, re.MULTILINE
    )
    if match:
        return int(match.group(1))
    return None


def parse_min_size(args_min_size: str | None) -> int | None:
    """Parse the --min-size argument into billion-of-parameters integer.

    Accepts values like '25B', '25', '3.5B', '3.5'.
    Returns the number of parameters in billions, or None if not set.
    """
    if not args_min_size:
        return None
    args_min_size = args_min_size.strip().upper()
    # Strip trailing 'B' if present
    numeric_str = args_min_size.rstrip("B")
    try:
        return int(float(numeric_str))
    except ValueError:
        print(
            f"WARNING: Invalid --min-size value '{args_min_size}'. "
            "Expected a number optionally followed by 'B' (e.g. '25B' or '25').",
            file=sys.stderr,
        )
        return None


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


def create_minimal_toml(
    model_name: str,
    framework: str,
    number_of_parameters: int,
    base_url: str = "http://localhost:1234",
) -> str:
    """Generate a minimal TOML config when no template is found.

    Uses number_of_parameters (in billions) so the config tracks model size.
    """
    return (
        f'name = "{model_name}"\n'
        f'base_url = "{base_url}"\n'
        f"temperature = 0.0\n"
        f"number_of_parameters = {number_of_parameters}\n"
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


def update_number_of_params_in_toml(
    content: str, number_of_parameters: int
) -> str:
    """Update (or insert) the number_of_parameters value in a TOML file content string."""
    if re.search(r"^number_of_parameters\s*=", content, re.MULTILINE):
        return re.sub(
            r"(number_of_parameters\s*=\s*)\d+",
            rf"\g<1>{number_of_parameters}",
            content,
        )
    # Insert number_of_parameters line after the name field (or at the top if no name)
    name_match = re.search(r"^name\s*=", content, re.MULTILINE)
    if name_match:
        insert_pos = name_match.end()
        return (
            content[:insert_pos]
            + f"\nnumber_of_parameters = {number_of_parameters}"
            + content[insert_pos:]
        )
    return content + f"\nnumber_of_parameters = {number_of_parameters}\n"


def read_toml_field(content: str, field: str) -> str | None:
    """Extract the value of a field from a TOML file content string."""
    match = re.search(
        rf"^{field}\s*=\s*[\"']?([^\"'\n#]+)", content, re.MULTILINE
    )
    if match:
        return match.group(1).strip().strip('"').strip("'")
    return None


def determine_runner_from_toml(toml_path: Path) -> str:
    """Infer the runner (ollama, lmstudio, or llama.cpp) from a model TOML's base_url.

    Ollama's default API port is 11434; LMStudio's is 1234; llama-server's
    default is 8080. Falls back to "lmstudio" when the file is missing or
    the base_url is unrecognised.
    """
    if not toml_path.is_file():
        return "lmstudio"
    base_url = read_toml_field(toml_path.read_text(), "base_url") or ""
    if "11434" in base_url:
        return "ollama"
    if f":{DEFAULT_LLAMACPP_PORT}" in base_url:
        return "llama.cpp"
    return "lmstudio"


def parse_lms_ls(output: str) -> list[tuple[str, str | None]]:
    """Parse the output of `lms ls` to extract model names and PARAMS values.

    Handles LMStudio's tabular output format. Returns a list of (model_name, params) tuples.
    Embedding models are excluded. PARAMS may be None if not present.
    """
    models: list[tuple[str, str | None]] = []
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
        if not match:
            continue

        model_name = match.group(1)

        # Extract PARAMS value (second column)
        # Find the position after the model name to get the rest of the line
        rest = stripped[match.end() :]
        params_match = re.match(r"(\S+)", rest)
        params_value = params_match.group(1) if params_match else None

        models.append((model_name, params_value))

    return models


def parse_params_value(params_str: str | None) -> int | None:
    """Parse a PARAMS value from `lms ls` into billion-of-parameters integer.

    Handles values like '7B', '3.5B', '7B-14B', etc.
    If the value has a hyphen, split at the hyphen and take the first value.
    Returns the parameter count in billions, or None if not parseable.
    """
    if params_str is None:
        return None
    params_str = params_str.strip().upper()
    if not params_str:
        return None
    # Handle hyphenated ranges (e.g., "7B-14B") - take the first value
    if "-" in params_str:
        params_str = params_str.split("-")[0]
    # Strip trailing 'B' if present
    numeric_str = params_str.rstrip("B")
    if not numeric_str:
        return None
    try:
        return int(float(numeric_str))
    except ValueError:
        return None


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


def load_model(model_name: str) -> bool:
    """Load a model in LM Studio. Returns True on success."""
    cmd = [
        "lms",
        "load",
        model_name,
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


def parse_ollama_ls(output: str) -> list[tuple[str, str | None]]:
    """Parse the output of `ollama ls` to extract model names and PARAMS values.

    Returns a list of (model_name, params) tuples.
    PARAMS may be None if not present.
    """
    models: list[tuple[str, str | None]] = []

    for line in output.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip header line
        if re.match(r"^NAME\s+ID\s+", stripped, re.IGNORECASE):
            continue

        # Skip summary line if present
        if "models" in stripped.lower() and "taking" in stripped.lower():
            continue

        # Extract model name (first column) and size/params (third column)
        # Format: "model:name    <id>    <size>    <modified>"
        parts = stripped.split()
        if len(parts) >= 3:
            model_name = parts[0]
            # The size field (e.g., "37 GB", "19 GB") - not params in billions
            # We'll use it as-is since ollama doesn't expose parameter counts directly
            size = parts[2] if len(parts) > 2 else None
            models.append((model_name, size))

    return models


def ollama_show_params(model_name: str) -> str | None:
    """Get the parameters value for an Ollama model via `ollama show`.

    Parses the `parameters` field under the `Model` section (e.g. "27.4B").
    Returns the raw string (e.g. "27.4B") or None if the command fails or
    the field is missing.
    """
    try:
        result = run_subprocess(["ollama", "show", model_name])
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        match = re.match(r"\s*parameters\s+(\S+)", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def parse_ollama_ps(output: str) -> list[str]:
    """Parse the output of `ollama ps` to extract only LOADED model names."""
    loaded = []

    for line in output.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Skip header line
        if re.match(r"^NAME\s+ID\s+", stripped, re.IGNORECASE):
            continue

        # Skip summary line if present
        if "models" in stripped.lower() and "taking" in stripped.lower():
            continue

        # Extract model name (first column)
        parts = stripped.split()
        if parts:
            loaded.append(parts[0])

    return loaded


def ollama_load_model(model_name: str) -> bool:
    """Pull/load a model in Ollama. Returns True on success."""
    cmd = [
        "ollama",
        "pull",
        model_name,
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


def ollama_unload_model(model_name: str) -> bool:
    """Unload a model from Ollama memory. Returns True on success."""
    cmd = ["ollama", "stop", model_name]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  Unloaded: {model_name}")
            return True
        else:
            # Model may not exist — that's OK
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


def ollama_unload_all_loaded() -> list[str]:
    """Unload all currently loaded models in Ollama. Returns list of unloaded model names."""
    try:
        result = run_subprocess(["ollama", "ps"])
        if result.returncode != 0:
            print(
                f"WARNING: 'ollama ps' failed ({result.returncode}), "
                "cannot determine loaded models.",
                file=sys.stderr,
            )
            return []
    except FileNotFoundError:
        print(
            "WARNING: 'ollama' not found, skipping preload cleanup.",
            file=sys.stderr,
        )
        return []

    loaded = parse_ollama_ps(result.stdout)
    if not loaded:
        return []

    unloaded = []
    for model in loaded:
        if ollama_unload_model(model):
            unloaded.append(model)
    return unloaded


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
# llama.cpp runner (uses llama-server with .gguf files from ~/.lmstudio/models)
# ---------------------------------------------------------------------------

# Tracks the currently running llama-server subprocess so we can stop it on
# unload. llama-server is a long-running OpenAI-compatible HTTP daemon; we
# spawn one per model and tear it down before loading the next.
_llamacpp_server_proc: subprocess.Popen | None = None


def _parse_params_from_gguf_name(name: str) -> str | None:
    """Extract a params token (e.g. "27B", "1.5B") from a GGUF filename.

    Returns the first plausible match like "27B" or "1.5B", or None.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    if match:
        return f"{match.group(1)}B"
    return None


def _gguf_model_id(gguf_path: Path, models_dir: Path) -> str:
    """Derive a model identifier from a GGUF file path.

    Returns just the filename stem (e.g. "Qwen3.6-27B-Q4_K_M") so that
    table rows show a short, readable model name rather than the full
    directory hierarchy from ~/.lmstudio/models.
    """
    return gguf_path.stem


def discover_gguf_files(
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
) -> list[Path]:
    """Find all *.gguf files under models_dir, skipping mmproj projector files.

    mmproj-*.gguf files are multimodal projectors and are not standalone
    LLMs — llama-server pairs them with their parent model via --mmproj.
    """
    if not models_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(models_dir.rglob("*.gguf")):
        if p.name.lower().startswith("mmproj"):
            continue
        out.append(p)
    return out


def discover_models_llamacpp(
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
) -> list[str]:
    """Discover all GGUF models under models_dir as model identifiers."""
    files = discover_gguf_files(models_dir)
    if not files:
        raise BenchmarkError(
            f"No .gguf models found under {models_dir}. "
            "Download a GGUF model in LM Studio first."
        )
    return [_gguf_model_id(p, models_dir) for p in files]


def discover_models_with_params_llamacpp(
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
) -> list[tuple[str, str | None]]:
    """Discover GGUF models with params parsed from their filenames."""
    files = discover_gguf_files(models_dir)
    if not files:
        raise BenchmarkError(
            f"No .gguf models found under {models_dir}. "
            "Download a GGUF model in LM Studio first."
        )
    return [
        (_gguf_model_id(p, models_dir), _parse_params_from_gguf_name(p.name))
        for p in files
    ]


def _resolve_gguf_path(
    model_id: str,
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
) -> Path | None:
    """Map a model identifier (GGUF stem) back to its on-disk .gguf path."""
    for p in discover_gguf_files(models_dir):
        if p.stem == model_id:
            return p
    return None


def _model_filter_str(
    model_name: str,
    runner: str,
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
) -> str:
    """Return the string to match --includes/--excludes against.

    For llama.cpp the short stem alone misses keywords that live in parent
    directory names (e.g. "MTP" in "unsloth/Qwen3.6-27B-MTP-GGUF/…").
    Resolving to the full relative path makes every path component searchable.
    For other runners the model name already contains all relevant tokens.
    """
    if runner == "llama.cpp":
        path = _resolve_gguf_path(model_name, models_dir)
        if path is not None:
            try:
                return str(path.relative_to(models_dir))
            except ValueError:
                return str(path)
    return model_name


def llamacpp_load_model(
    model_id: str,
    models_dir: Path = DEFAULT_LLAMACPP_MODELS_DIR,
    port: int = DEFAULT_LLAMACPP_PORT,
) -> bool:
    """Start a llama-server process for the given GGUF model.

    Stores the Popen handle in _llamacpp_server_proc so unload can stop it.
    Returns True if the process started; readiness is checked separately by
    verify_model_loaded.
    """
    global _llamacpp_server_proc

    # Ensure any previous server is stopped before launching a new one.
    llamacpp_unload_model(model_id)

    gguf_path = _resolve_gguf_path(model_id, models_dir)
    if gguf_path is None:
        print(
            f"  ERROR: could not resolve {model_id} to a .gguf file under "
            f"{models_dir}",
            file=sys.stderr,
        )
        return False

    cmd = [
        "llama-server",
        "-m",
        str(gguf_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "999",
        "-a",
        model_id,
    ]
    print(f"  Launching: {' '.join(cmd)}")
    try:
        _llamacpp_server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(
            "  ERROR: 'llama-server' not found. Install llama.cpp "
            "(e.g. `brew install llama.cpp`).",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"  ERROR launching llama-server for {model_id}: {e}", file=sys.stderr)
        return False

    print(f"  Loaded: {model_id} (pid {_llamacpp_server_proc.pid})")
    return True


def llamacpp_unload_model(model_id: str | None = None) -> bool:
    """Stop the running llama-server process, if any."""
    global _llamacpp_server_proc
    proc = _llamacpp_server_proc
    if proc is None:
        return True
    if proc.poll() is not None:
        _llamacpp_server_proc = None
        return True
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        label = model_id or f"pid {proc.pid}"
        print(f"  Unloaded: {label}")
    except Exception as e:
        print(
            f"  WARNING unloading llama-server ({model_id}): {e}",
            file=sys.stderr,
        )
        return False
    finally:
        _llamacpp_server_proc = None
    return True


def llamacpp_unload_all_loaded() -> list[str]:
    """Stop the tracked llama-server (we only ever spawn one at a time)."""
    if _llamacpp_server_proc is None or _llamacpp_server_proc.poll() is not None:
        return []
    if llamacpp_unload_model():
        return ["llama-server"]
    return []


def verify_model_loaded(
    model_name: str,
    base_url: str,
    timeout: float = DEFAULT_LOAD_VERIFY_TIMEOUT,
    retry_interval: float = LOAD_VERIFY_RETRY_INTERVAL,
) -> bool:
    """Probe the model with a tiny chat-completions request until it responds.

    `lms load` sometimes returns success before the model is queryable, and
    `ollama pull` only downloads weights without loading them into VRAM —
    the first benchmark request then times out or errors. Polling the
    OpenAI-compatible endpoint until a 200 comes back catches both cases
    and lets us skip the model cleanly instead of poisoning the run.
    """
    if not base_url:
        print(
            f"  WARNING: no base_url available for {model_name}; "
            "skipping load verification.",
            file=sys.stderr,
        )
        return True

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    deadline = time.monotonic() + timeout
    last_error: str | None = None
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        remaining = max(1.0, deadline - time.monotonic())
        try:
            with httpx.Client(timeout=remaining) as client:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code == 200:
                    print(
                        f"  Verified loaded: {model_name} "
                        f"(attempt {attempt})"
                    )
                    return True
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        if time.monotonic() >= deadline:
            break
        time.sleep(retry_interval)

    print(
        f"  ERROR: {model_name} not responding after {timeout:.0f}s "
        f"({attempt} attempt(s)). Last error: {last_error}",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark suite orchestrator for LMStudio, Ollama, and llama.cpp models. "
            "Discovers models from each runner, runs bench.py sequentially, and "
            "produces Markdown + HTML results tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Core ----------------------------------------------------------------
    parser.add_argument(
        "--corpus",
        required=True,
        help=(
            "Corpus to benchmark against. Either a named config under "
            f"{DEFAULT_CORPORA_DIR}/ (e.g. 'http_server') or a direct path to "
            "a Python source file."
        ),
    )
    parser.add_argument(
        "--runner",
        "--runners",
        dest="runner",
        type=str,
        default="ollama,lmstudio,llama.cpp",
        help=(
            "Comma-delimited list of runners to use "
            "(default: ollama,lmstudio,llama.cpp). "
            "Choices: ollama, lmstudio, llama.cpp. "
            "Example: --runner llama.cpp,ollama"
        ),
    )

    # ---- Filtering -----------------------------------------------------------
    parser.add_argument(
        "--min-size",
        type=str,
        default=None,
        metavar="SIZE",
        help=(
            "Skip models smaller than SIZE billion parameters. "
            "Accepts '25B', '25', '3.5B', etc. "
            "For lmstudio/ollama the value comes from `lms ls` / `ollama show`; "
            "for llama.cpp it is parsed from the GGUF filename."
        ),
    )
    parser.add_argument(
        "--includes",
        type=str,
        default=None,
        metavar="KW[,KW...]",
        help=(
            "Comma-delimited keywords (case-insensitive). Only models whose name "
            "contains ANY keyword are benchmarked. For llama.cpp the full path "
            "under ~/.lmstudio/models is searched, so directory components such "
            "as 'MTP' or 'unsloth' are also matchable."
        ),
    )
    parser.add_argument(
        "--excludes",
        type=str,
        default=None,
        metavar="KW[,KW...]",
        help=(
            "Comma-delimited keywords (case-insensitive). Models whose name "
            "contains ANY keyword are skipped. Same path-aware matching as "
            "--includes for llama.cpp models."
        ),
    )
    parser.add_argument(
        "--show",
        type=str,
        default=None,
        metavar="KW[,KW...]",
        help=(
            "Comma-delimited keywords (case-insensitive). Only models whose "
            "name contains ANY keyword are shown in the results table; all "
            "others are hidden. Same path-aware matching as --includes for "
            "llama.cpp models."
        ),
    )

    # ---- Run behaviour -------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken without loading models or running benchmarks.",
    )
    parser.add_argument(
        "--clean-run",
        action="store_true",
        help=(
            "Delete the existing results table and all JSON result files for "
            "the chosen corpus before starting, so the run produces a fresh table."
        ),
    )
    parser.add_argument(
        "--recreate-table",
        "--recreate_table",
        dest="recreate_table",
        action="store_true",
        help=(
            "Skip model discovery and benchmarks entirely. Re-read all existing "
            "JSON result files in results/ for the chosen corpus and regenerate "
            "the Markdown and HTML tables from them."
        ),
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        metavar="N",
        help=f"Seconds to wait between consecutive model benchmarks (default: {DEFAULT_COOLDOWN_SECONDS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_SUBPROCESS_TIMEOUT,
        metavar="SECS",
        help=(
            f"Maximum wall-clock seconds allowed for a single model's benchmark "
            f"subprocess (default: {DEFAULT_SUBPROCESS_TIMEOUT})."
        ),
    )
    parser.add_argument(
        "--load-verify-timeout",
        type=float,
        default=DEFAULT_LOAD_VERIFY_TIMEOUT,
        metavar="SECS",
        help=(
            f"Seconds to poll the model's /v1/chat/completions endpoint after "
            f"loading, waiting for it to become ready (default: {DEFAULT_LOAD_VERIFY_TIMEOUT}). "
            "Set to 0 to skip verification."
        ),
    )

    # ---- Paths ---------------------------------------------------------------
    parser.add_argument(
        "--save-table",
        type=Path,
        default=DEFAULT_SAVE_TABLE,
        metavar="PATH",
        help=f"Where to write the Markdown results table (default: {DEFAULT_SAVE_TABLE}). An HTML version is written alongside it.",
    )
    parser.add_argument(
        "--corpora-dir",
        type=Path,
        default=DEFAULT_CORPORA_DIR,
        metavar="DIR",
        help=f"Directory containing corpus TOML configs (default: {DEFAULT_CORPORA_DIR}).",
    )
    parser.add_argument(
        "--user-models-dir",
        type=Path,
        default=DEFAULT_USER_MODELS_DIR,
        metavar="DIR",
        help=f"Directory where per-run model TOML configs are written (default: {DEFAULT_USER_MODELS_DIR}).",
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        metavar="DIR",
        help=f"Directory containing model template TOMLs used as a base for discovered models (default: {DEFAULT_TEMPLATE_DIR}).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def discover_models(runner: str = "lmstudio") -> list[str]:
    """Discover all installed models via `lms ls` or `ollama ls`."""
    if runner == "ollama":
        return discover_models_ollama()
    if runner == "llama.cpp":
        return discover_models_llamacpp()
    return discover_models_lmstudio()


def discover_models_lmstudio() -> list[str]:
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

    parsed = parse_lms_ls(result.stdout)
    if not parsed:
        raise BenchmarkError(
            "No models found via 'lms ls'. Install a model in LM Studio first."
        )

    return [m[0] for m in parsed]


def discover_models_ollama() -> list[str]:
    """Discover all installed models via `ollama ls`."""
    try:
        result = run_subprocess(["ollama", "ls"])
    except FileNotFoundError:
        raise BenchmarkError(
            "'ollama' executable not found. Is Ollama installed and on PATH?"
        )

    if result.returncode != 0:
        raise BenchmarkError(
            f"'ollama ls' failed with code {result.returncode}: {result.stderr}"
        )

    parsed = parse_ollama_ls(result.stdout)
    if not parsed:
        raise BenchmarkError(
            "No models found via 'ollama ls'. Install a model with 'ollama pull' first."
        )

    return [m[0] for m in parsed]


def discover_models_with_params(
    runner: str = "lmstudio",
) -> list[tuple[str, str | None]]:
    """Discover all installed models, returning (model_name, params) tuples."""
    if runner == "ollama":
        return discover_models_with_params_ollama()
    if runner == "llama.cpp":
        return discover_models_with_params_llamacpp()
    return discover_models_with_params_lmstudio()


def discover_models_with_params_lmstudio() -> list[tuple[str, str | None]]:
    """Discover all installed models via `lms ls`, returning (model_name, params) tuples."""
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

    parsed = parse_lms_ls(result.stdout)
    if not parsed:
        raise BenchmarkError(
            "No models found via 'lms ls'. Install a model in LM Studio first."
        )

    return parsed


def discover_models_with_params_ollama() -> list[tuple[str, str | None]]:
    """Discover all installed models via `ollama ls`, returning (model_name, params) tuples.

    `ollama ls` only exposes the on-disk size (e.g. "19 GB"), which is not the
    parameter count. To support `--min-size` filtering we look up each model
    via `ollama show` and parse the `parameters` field (e.g. "27.4B").
    """
    try:
        result = run_subprocess(["ollama", "ls"])
    except FileNotFoundError:
        raise BenchmarkError(
            "'ollama' executable not found. Is Ollama installed and on PATH?"
        )

    if result.returncode != 0:
        raise BenchmarkError(
            f"'ollama ls' failed with code {result.returncode}: {result.stderr}"
        )

    parsed = parse_ollama_ls(result.stdout)
    if not parsed:
        raise BenchmarkError(
            "No models found via 'ollama ls'. Install a model with 'ollama pull' first."
        )

    enriched: list[tuple[str, str | None]] = []
    for model_name, _ in parsed:
        enriched.append((model_name, ollama_show_params(model_name)))
    return enriched


def generate_user_configs(
    models: list[str],
    template_dir: Path,
    user_models_dir: Path,
    dry_run: bool = False,
    runner: str = "lmstudio",
) -> list[Path]:
    """Create a user config TOML for each model.

    Returns the list of created config file paths.
    """
    user_models_dir.mkdir(parents=True, exist_ok=True)

    if runner == "ollama":
        base_url = DEFAULT_OLLAMA_BASE_URL
    elif runner == "llama.cpp":
        base_url = DEFAULT_LLAMACPP_BASE_URL
    else:
        base_url = "http://localhost:1234"

    created: list[Path] = []
    seen_filenames: dict[str, Path] = {}  # For collision detection

    for model_name in models:
        framework = determine_framework(model_name)
        safe_name = sanitize_model_name(model_name)

        toml_filename = f"{safe_name}-{framework}.toml"

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
            # Update number_of_parameters from template (or add if missing)
            model_params = parse_model_params(template)
            if model_params is not None:
                content = update_number_of_params_in_toml(content, model_params)
            # Bug #3 fix: update the name field to the discovered model id
            content = update_toml_name_field(content, model_name)
            # Update base_url for the runner
            content = re.sub(
                r"^(base_url\s*=\s*).+",
                rf'\g<1>"{base_url}"',
                content,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            print(
                f"  (no template found for {model_name}, generating minimal config)"
            )
            # Default to 0 params (unknown) when no template exists
            content = create_minimal_toml(model_name, framework, 0, base_url)

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
    runner: str = "lmstudio",
    load_verify_timeout: float = DEFAULT_LOAD_VERIFY_TIMEOUT,
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
      - runner: name of the runner used (lmstudio / ollama)
    """
    results: list[dict] = []

    if runner == "ollama":
        load_model_fn = ollama_load_model
        unload_model_fn = ollama_unload_model
    elif runner == "llama.cpp":
        load_model_fn = llamacpp_load_model
        unload_model_fn = llamacpp_unload_model
    else:
        load_model_fn = load_model
        unload_model_fn = unload_model

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
                    "runner": runner,
                }
            )
            continue

        # Load the model before benchmarking
        if not load_model_fn(model_name):
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
                    "runner": runner,
                }
            )
            # Sleep between benchmarks (not after the last one)
            if i < len(config_paths) - 1:
                print(f"  Sleeping {cooldown_seconds} seconds...")
                time.sleep(cooldown_seconds)
            continue

        # Verify the model is actually serving requests before launching the
        # benchmark subprocess. `lms load` / `ollama pull` can succeed before
        # the model is queryable, which causes bench.py to fail mid-corpus.
        if load_verify_timeout > 0:
            base_url = (
                read_toml_field(config_path.read_text(), "base_url") or ""
            )
            if not verify_model_loaded(
                model_name, base_url, timeout=load_verify_timeout
            ):
                print(
                    f"  SKIPPED: {model_name} loaded but did not become ready.",
                    file=sys.stderr,
                )
                unload_model_fn(model_name)
                results.append(
                    {
                        "config_path": config_path,
                        "passed": 0,
                        "hallucinated": 0,
                        "bonus": 0,
                        "runtime": 0.0,
                        "error": "model not ready after load",
                        "runner": runner,
                    }
                )
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
                "--timeout",
                str(subprocess_timeout),
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
            unload_model_fn(model_name)

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

        # `completion_tokens` is written by newer runner.py dumps. Older dumps
        # only have the raw response text, so count retro-actively with tiktoken
        # — same encoder, comparable across rows.
        completion_tokens = 0
        for r in results:
            tokens = r.get("completion_tokens")
            if tokens is None:
                tokens = count_tokens(r.get("response", "") or "")
            completion_tokens += int(tokens)
        tokens_per_sec = (
            completion_tokens / runtime if runtime > 0 else 0.0
        )

        # JSON dump files are named {corpus}__{model-stem}.json.
        # Extract the model-stem (part after '__') and reconstruct
        # the config path in user_models_dir.
        model_stem = (
            json_file.stem.split("__")[1]
            if "__" in json_file.stem
            else json_file.stem
        )

        config_path = user_models_dir / f"{model_stem}.toml"
        parsed_results.append(
            {
                "config_path": config_path,
                "passed": passed,
                "hallucinated": hallucinated,
                "bonus": bonus,
                "primary_matched": primary_matched,
                "runtime": runtime,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": tokens_per_sec,
                "error": None,
                "runner": determine_runner_from_toml(config_path),
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

    # Overwrite with existing (parsed) data — existing takes priority.
    # Only update entries already in new_map so we don't absorb results
    # from other runners (parse_results_from_files scans all corpus JSON
    # files, not just the current runner's).
    for r in existing:
        cp = str(r.get("config_path", ""))
        if cp in new_map:
            new_map[cp] = r

    return list(new_map.values())


def _build_table_data(
    results: list[dict],
    show_keywords: list[str] | None = None,
) -> dict:
    """Build shared table data from benchmark results."""
    # Filter results by show keywords if provided
    if show_keywords:
        filtered = []
        for r in results:
            config_path = r.get("config_path", Path("unknown"))
            model = config_path.stem
            if isinstance(config_path, Path) and config_path.is_file():
                toml_name = read_toml_field(config_path.read_text(), "name")
                if toml_name:
                    model = toml_name.rsplit("/", 1)[-1]
            model_lower = model.lower()
            if any(kw in model_lower for kw in show_keywords):
                filtered.append(r)
        sorted_results = sorted(
            filtered,
            key=lambda r: (-r.get("passed", 0), r.get("hallucinated", 0)),
        )
    else:
        sorted_results = sorted(
            results,
            key=lambda r: (-r.get("passed", 0), r.get("hallucinated", 0)),
        )

    header = [
        "Model",
        "Runner",
        "Pass",
        "Hallucinations",
        "Bonus",
        "Primary",
        "Runtime (s)",
        "Tokens",
        "Tokens/s",
    ]
    rows: list[list[str]] = []

    total_passed = 0
    total_hallucinated = 0
    total_bonus = 0
    total_primary = 0
    total_runtime = 0.0
    total_tokens = 0

    for r in sorted_results:
        config_path = r.get("config_path", Path("unknown"))
        model = config_path.stem
        runner = r.get("runner")
        if not runner and isinstance(config_path, Path):
            runner = determine_runner_from_toml(config_path)
        if not runner:
            runner = "lmstudio"
        # For llama.cpp, read the TOML's name field and take only its last
        # path component so the table always shows a short model name
        # regardless of whether the JSON was created before or after the
        # short-name change (old names were full relative paths like
        # "lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M").
        if runner == "llama.cpp" and isinstance(config_path, Path) and config_path.is_file():
            toml_name = read_toml_field(config_path.read_text(), "name")
            if toml_name:
                model = toml_name.rsplit("/", 1)[-1]
        passed = r.get("passed", 0)
        hallucinated = r.get("hallucinated", 0)
        bonus = r.get("bonus", 0)
        primary = r.get("primary_matched", 0)
        runtime = r.get("runtime", 0.0)
        tokens = r.get("completion_tokens", 0)
        tokens_per_sec = r.get("tokens_per_sec", 0.0)
        error = r.get("error")

        if error:
            model = f"{model} (ERROR: {error})"

        rows.append(
            [
                model,
                runner,
                str(passed),
                str(hallucinated),
                str(bonus),
                str(primary),
                f"{runtime:.1f}",
                f"{tokens:,}" if tokens else "-",
                f"{tokens_per_sec:.1f}" if tokens_per_sec else "-",
            ]
        )

        total_passed += passed
        total_hallucinated += hallucinated
        total_bonus += bonus
        total_primary += primary
        total_runtime += runtime
        total_tokens += tokens

    return {
        "header": header,
        "rows": rows,
        "corpus": "",
    }


def generate_markdown_table(
    results: list[dict],
    save_path: Path,
    corpus: str = "",
    show_keywords: list[str] | None = None,
) -> None:
    """Generate a Markdown table with improved formatting.

    Appends new rows between markers so previous runs are preserved.
    Uses equal-width columns and one row per model for readability.
    """
    data = _build_table_data(results, show_keywords=show_keywords)
    data["corpus"] = corpus

    header = data["header"]
    rows = data["rows"]

    # Compute max width per column
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
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
            header_line = _fmt_row(header)
            separator_line = _fmt_separator()
            outer_preamble = (
                f"<!-- Corpus: {corpus} -->"
                + "\n\n"
                + header_line
                + "\n"
                + separator_line
                + "\n"
                + marker_start
            )
            after = existing[end_idx:]
            table_content = outer_preamble + new_block + after
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
    corpus: str,
    show_keywords: list[str] | None = None,
) -> str:
    """Build a complete HTML file from scratch."""
    title = f"Results Table{f' — {corpus}' if corpus else ''}"
    toggle_buttons = "".join(
        f'<button class="col-toggle active" data-col="{i}">{h}</button>'
        for i, h in enumerate(header)
    )
    show_kw_json = json.dumps(show_keywords or [])
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
            padding: 16px;
            margin: 0;
            box-sizing: border-box;
        }}
        .table-container {{
            width: 100%;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow-x: auto;
        }}
        h2 {{
            margin: 0;
            padding: 16px 24px;
            background: #1a73e8;
            color: #fff;
            font-size: 18px;
            font-weight: 500;
            border-radius: 8px 8px 0 0;
        }}
        .col-toggle-bar {{
            padding: 8px 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }}
        .col-toggle-label {{
            font-size: 12px;
            color: #666;
            font-weight: 500;
            margin-right: 2px;
            white-space: nowrap;
        }}
        .col-toggle {{
            padding: 3px 10px;
            border: 1px solid #1a73e8;
            border-radius: 12px;
            background: #1a73e8;
            color: #fff;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.15s, color 0.15s;
            line-height: 1.4;
        }}
        .col-toggle:not(.active) {{
            background: #fff;
            color: #1a73e8;
        }}
        table {{
            width: 100%;
            min-width: 600px;
            border-collapse: collapse;
            table-layout: auto;
        }}
        thead th {{
            background: #f8f9fa;
            padding: 10px 24px 10px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            color: #555;
            border-bottom: 2px solid #e0e0e0;
            cursor: pointer;
            user-select: none;
            position: relative;
            white-space: nowrap;
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
        .resize-handle {{
            position: absolute;
            top: 0;
            right: 0;
            width: 6px;
            height: 100%;
            cursor: col-resize;
            user-select: none;
            background: transparent;
        }}
        .resize-handle:hover, .resize-handle.dragging {{
            background: rgba(26, 115, 232, 0.4);
        }}
        body.col-resizing {{
            cursor: col-resize !important;
            user-select: none !important;
        }}
    </style>
</head>
<body>
    <div class="table-container">
        <h2>{title}</h2>
        <div class="col-toggle-bar">
            <span class="col-toggle-label">Columns:</span>
            {toggle_buttons}
        </div>
        <div class="col-toggle-bar" id="model-filter-bar">
            <span class="col-toggle-label">Models:</span>
            <input type="text" id="model-filter-input" placeholder="Filter models..." style="padding:3px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px;width:200px;" />
        </div>
        <table>
            <thead>
                <tr>
{"".join(header_cells)}
                </tr>
            </thead>
            <tbody id="benchmark-rows">
{"".join(html_rows)}
            </tbody><!-- BENCHMARK_ROWS_END -->
        </table>
    </div>
    <script>
    (function() {{
        const SHOW_KEYWORDS = {show_kw_json};
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
                return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
            }});
            tbody.innerHTML = rows.map(r => r.outerHTML).join("\\n");
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

        function setColumnWidth(colIndex, width) {{
            const px = Math.max(40, width) + "px";
            const th = document.querySelectorAll("thead th")[colIndex];
            if (th) th.style.minWidth = px;
        }}

        function attachResize(th, colIndex) {{
            const handle = document.createElement("div");
            handle.className = "resize-handle";
            th.appendChild(handle);
            let startX = 0, startWidth = 0, dragging = false;
            handle.addEventListener("mousedown", (e) => {{
                e.preventDefault();
                e.stopPropagation();
                dragging = true;
                startX = e.pageX;
                startWidth = th.offsetWidth;
                handle.classList.add("dragging");
                document.body.classList.add("col-resizing");
                const onMove = (ev) => {{
                    if (!dragging) return;
                    setColumnWidth(colIndex, startWidth + (ev.pageX - startX));
                }};
                const onUp = () => {{
                    dragging = false;
                    handle.classList.remove("dragging");
                    document.body.classList.remove("col-resizing");
                    document.removeEventListener("mousemove", onMove);
                    document.removeEventListener("mouseup", onUp);
                }};
                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
            }});
            handle.addEventListener("click", (e) => e.stopPropagation());
        }}

        function toggleColumn(colIndex, visible) {{
            const th = document.querySelectorAll("thead th")[colIndex];
            if (th) th.style.display = visible ? "" : "none";
            document.querySelectorAll(
                `#benchmark-rows tr td:nth-child(${{colIndex + 1}})`
            ).forEach(td => {{ td.style.display = visible ? "" : "none"; }});
        }}

        function applyModelFilter() {{
            const input = document.getElementById("model-filter-input");
            if (!input) return;
            const kw = input.value.trim().toLowerCase();
            const rows = document.querySelectorAll("#benchmark-rows tr");
            rows.forEach(row => {{
                const modelCell = row.querySelector("td:first-child");
                if (!modelCell) return;
                const modelName = modelCell.textContent.toLowerCase();
                if (!kw) {{
                    row.style.display = "";
                    return;
                }}
                const match = SHOW_KEYWORDS.length > 0
                    ? SHOW_KEYWORDS.some(k => modelName.includes(k))
                    : modelName.includes(kw);
                row.style.display = match ? "" : "none";
            }});
        }}

        document.getElementById("model-filter-input")?.addEventListener("input", applyModelFilter);

        document.querySelectorAll("thead th").forEach((th, i) => {{
            th.addEventListener("click", (e) => {{
                if (e.target.classList.contains("resize-handle")) return;
                sortTable(i);
            }});
            const arrow = document.createElement("span");
            arrow.className = "sort-arrow";
            arrow.textContent = " \\u25B2\\u25BC";
            th.appendChild(arrow);
            attachResize(th, i);
        }});

        document.querySelectorAll(".col-toggle").forEach((btn) => {{
            btn.addEventListener("click", () => {{
                const col = parseInt(btn.dataset.col, 10);
                const nowActive = btn.classList.toggle("active");
                toggleColumn(col, nowActive);
            }});
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
    show_keywords: list[str] | None = None,
) -> None:
    """Generate an HTML table with the same data as the Markdown table.

    Appends new rows between markers so previous runs are preserved.
    """
    data = _build_table_data(results, show_keywords=show_keywords)
    data["corpus"] = corpus

    header = data["header"]
    rows = data["rows"]

    html_rows = []
    for row in rows:
        cells = [
            f'<td style="padding: 6px 12px; text-align: left; '
            f'border-bottom: 1px solid #ddd; vertical-align: top;">{cell}</td>'
            for cell in row
        ]
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    header_cells = [
        f'<th data-column="{i}">{h}</th>'
        for i, h in enumerate(header)
    ]

    save_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = _build_full_html(
        header, header_cells, html_rows, corpus, show_keywords
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

    # Parse comma-delimited runners
    runners: list[str] = [
        r.strip() for r in args.runner.split(",") if r.strip()
    ]
    if not runners:
        print("No runners specified.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  Benchmark Suite — Multi-Runner Orchestrator")
    print("=" * 60)
    print(f"  Runners:         {', '.join(runners)}")
    print(f"  Corpus:          {args.corpus}")
    print(f"  Min size (B):    {args.min_size or '(all)'}")
    print(f"  Corpora dir:     {args.corpora_dir}")
    print(f"  User models:     {args.user_models_dir}")
    print(f"  Template dir:    {args.template_dir}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"  Clean run:       {args.clean_run}")
    print(f"  Save table:      {args.save_table}")
    print(f"  Cooldown (s):    {args.cooldown_seconds}")
    print(f"  Timeout (s):     {args.timeout}")
    print(f"  Load verify (s): {args.load_verify_timeout}")
    includes_list = (
        [k.strip().lower() for k in args.includes.split(",") if k.strip()]
        if args.includes
        else []
    )
    excludes_list = (
        [k.strip().lower() for k in args.excludes.split(",") if k.strip()]
        if args.excludes
        else []
    )
    show_list = (
        [k.strip().lower() for k in args.show.split(",") if k.strip()]
        if args.show
        else []
    )
    print(
        f"  Includes:        {', '.join(includes_list) if includes_list else '(all)'}"
    )
    print(
        f"  Excludes:        {', '.join(excludes_list) if excludes_list else '(none)'}"
    )
    print(
        f"  Show (table):    {', '.join(show_list) if show_list else '(all)'}"
    )
    print("=" * 60)

    # Parse --show keywords for table filtering
    show_keywords = (
        [k.strip().lower() for k in args.show.split(",") if k.strip()]
        if args.show
        else []
    )

    # Ensure all relevant directories exist
    args.corpora_dir.mkdir(parents=True, exist_ok=True)
    args.template_dir.mkdir(parents=True, exist_ok=True)
    args.user_models_dir.mkdir(parents=True, exist_ok=True)

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
            print(
                f"  Removed {removed} result file(s) for corpus '{args.corpus}'."
            )

    # --recreate-table: skip discovery/benchmarks, just rebuild the tables
    # from existing JSON dumps in results/.
    if args.recreate_table:
        print("\n[Recreate] Rebuilding results tables from existing JSON dumps...")
        parsed = parse_results_from_files(
            args.corpus, args.user_models_dir, Path("results")
        )
        if not parsed:
            print(
                f"  No JSON results found for corpus '{args.corpus}' in results/. "
                "Run a benchmark first.",
                file=sys.stderr,
            )
            return 1
        print(f"  Found {len(parsed)} result file(s).")
        generate_markdown_table(parsed, args.save_table, args.corpus)
        html_table_path = args.save_table.with_suffix(".html")
        generate_html_table(parsed, html_table_path, args.corpus)
        if not args.dry_run:
            print("\n[Recreate] Generating visual report...")
            run_visualization()
        print("\n" + "=" * 60)
        print("  Done!")
        print("=" * 60)
        return 0

    all_results: list[dict] = []

    for runner in runners:
        if runner == "ollama":
            runner_label = "Ollama"
            discover_cmd = "ollama ls"
        elif runner == "llama.cpp":
            runner_label = "llama.cpp"
            discover_cmd = f"scan {DEFAULT_LLAMACPP_MODELS_DIR}"
        else:
            runner_label = "LMStudio"
            discover_cmd = "lms ls"

        print("\n" + "=" * 60)
        print(f"  Runner: {runner_label}")
        print("=" * 60)

        try:
            # Step 1: Discover models
            print(
                f"\n[1/7] Discovering {runner_label} models via `{discover_cmd}`..."
            )
            models = discover_models(runner=runner)

            if not models:
                print(
                    f"  No models found via `{discover_cmd}`; skipping {runner_label}.",
                    file=sys.stderr,
                )
                continue

            # Apply --includes filter
            if includes_list:
                filtered = []
                for m in models:
                    m_lower = _model_filter_str(m, runner).lower()
                    if any(kw in m_lower for kw in includes_list):
                        filtered.append(m)
                models = filtered
                if not models:
                    print(
                        f"  No models match includes {[', '.join(includes_list)]}; skipping {runner_label}.",
                        file=sys.stderr,
                    )
                    continue

            # Apply --excludes filter
            if excludes_list:
                filtered = []
                for m in models:
                    m_lower = _model_filter_str(m, runner).lower()
                    if not any(kw in m_lower for kw in excludes_list):
                        filtered.append(m)
                models = filtered
                if not models:
                    print(
                        f"  All models excluded by {[', '.join(excludes_list)]}; skipping {runner_label}.",
                        file=sys.stderr,
                    )
                    continue

            # Apply --min-size filter based on PARAMS from `lms ls` / `ollama ls`
            min_size = parse_min_size(args.min_size)
            if min_size is not None:
                # Use discover_models_with_params to get PARAMS values
                models_with_params = discover_models_with_params(runner=runner)
                filtered = []
                for model_name, params_str in models_with_params:
                    if model_name not in models:
                        continue
                    parsed_params = parse_params_value(params_str)
                    if parsed_params is not None and parsed_params >= min_size:
                        filtered.append(model_name)
                    elif parsed_params is None:
                        # If no params from ls, include it (can't filter)
                        ls_cmd = "ollama ls" if runner == "ollama" else "lms ls"
                        print(
                            f"  WARNING: No PARAMS value for {model_name} in `{ls_cmd}` output, including by default.",
                            file=sys.stderr,
                        )
                        filtered.append(model_name)
                models = filtered
                if not models:
                    print(
                        f"  No models meet the --min-size {args.min_size} threshold; skipping {runner_label}.",
                        file=sys.stderr,
                    )
                    continue
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
                dry_run=args.dry_run,
                runner=runner,
            )

            if not config_paths:
                print("No configs generated. Exiting.", file=sys.stderr)
                return 1

            # Step 3.5: Unload all loaded models so we always load one at a time
            print("\n[3.5/7] Unloading all loaded models...")
            if runner == "ollama":
                unloaded = ollama_unload_all_loaded()
            elif runner == "llama.cpp":
                unloaded = llamacpp_unload_all_loaded()
            else:
                unloaded = unload_all_loaded()
            if unloaded:
                print(
                    f"  Unloaded {len(unloaded)} model(s) to ensure clean state."
                )
            else:
                print("  No models were loaded (clean state).")

            # Step 4: Run benchmarks (with per-model load/unload)
            print("\n[4/7] Running benchmarks...")
            benchmark_results = run_benchmarks(
                config_paths=config_paths,
                corpus=args.corpus,
                cooldown_seconds=args.cooldown_seconds,
                dry_run=args.dry_run,
                subprocess_timeout=args.timeout,
                runner=runner,
                load_verify_timeout=args.load_verify_timeout,
            )

            # Step 5: Parse results from JSON dump files (if any were generated)
            print("\n[5/7] Parsing benchmark results...")
            parsed_results = parse_results_from_files(
                args.corpus, args.user_models_dir, Path("results")
            )

            # Merge parsed results with benchmark results (by config_path)
            benchmark_results = merge_results(parsed_results, benchmark_results)

            # Collect results for this runner
            all_results.extend(benchmark_results)

            # Step 6: Generate Markdown table
            print("\n[6/7] Generating Markdown results table...")
            generate_markdown_table(
                benchmark_results, args.save_table, args.corpus,
                show_keywords=show_keywords or None,
            )

            # Step 6b: Generate HTML table
            html_table_path = args.save_table.with_suffix(".html")
            print("\n[6b/7] Generating HTML results table...")
            generate_html_table(
                benchmark_results, html_table_path, args.corpus,
                show_keywords=show_keywords or None,
            )

            # Step 7: Run visualization (if not dry-run)
            if not args.dry_run:
                print("\n[7/7] Generating visual report...")
                run_visualization()

        except BenchmarkError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1

    # Generate consolidated results table across all runners. Re-parse every
    # JSON dump for this corpus so previous runs are preserved when the user
    # did not pass --clean-run (which would have wiped the JSON files at
    # startup). Current-run entries without a JSON file (load failures,
    # timeouts) are merged in so errors stay visible.
    all_parsed = parse_results_from_files(
        args.corpus, args.user_models_dir, Path("results")
    )
    by_cp: dict[str, dict] = {}
    for r in all_results:
        cp = str(r.get("config_path", ""))
        if cp:
            by_cp[cp] = r
    for r in all_parsed:
        cp = str(r.get("config_path", ""))
        if cp:
            by_cp[cp] = r
    final_results = list(by_cp.values())

    if final_results:
        print("\n" + "=" * 60)
        print("  Consolidated Results (All Runners)")
        print("=" * 60)
        print("\n[Final] Generating consolidated Markdown results table...")
        generate_markdown_table(
            final_results, args.save_table, args.corpus,
            show_keywords=show_keywords or None,
        )

        html_table_path = args.save_table.with_suffix(".html")
        print("\n[Final] Generating consolidated HTML results table...")
        generate_html_table(
            final_results, html_table_path, args.corpus,
            show_keywords=show_keywords or None,
        )

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
