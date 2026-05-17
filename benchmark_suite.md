# Benchmark Suite

Benchmark suite orchestrator for **LMStudio**, **Ollama**, and **llama.cpp** models. Automates the full pipeline: discovering installed models, generating per-model config files, running sequential benchmarks, and producing results tables with visual reports.

## What It Does

1. **Discovers models** from each runner:
   - LMStudio — via `lms ls`
   - Ollama — via `ollama ls`
   - llama.cpp — by scanning `~/.lmstudio/models` for `.gguf` files
2. **Generates per-model TOML configs** from templates in `configs/models/`, injecting the correct `base_url` for the selected runner.
3. **Runs benchmarks** sequentially — loading each model, running `bench.py run`, then unloading it.
4. **Produces a Markdown results table** with per-model pass rate, hallucinated lines, bonus lines, runtime, token count, and tokens/sec.
5. **Generates an HTML results table** with sortable columns, resizable columns, and per-column visibility toggles.
6. **Runs a visualization script** (`analysis/visualize.py`) to produce interactive HTML dashboards.

## Prerequisites

- [pixi](https://pixi.sh/) installed
- One or more runners available:
  - **LMStudio**: `lms` CLI on PATH, local server running (port 1234)
  - **Ollama**: `ollama` on PATH and running (port 11434)
  - **llama.cpp**: `llama-server` on PATH (e.g. `brew install llama.cpp`), `.gguf` models under `~/.lmstudio/models`
- Corpus TOMLs in `configs/corpora/` or a direct path to a source file

## Runners

| Runner      | Discovery              | Load / Unload                     | API port |
|-------------|------------------------|-----------------------------------|----------|
| `lmstudio`  | `lms ls`               | `lms load` / `lms unload`         | 1234     |
| `ollama`    | `ollama ls`            | `ollama pull` / `ollama stop`     | 11434    |
| `llama.cpp` | scan `~/.lmstudio/models/*.gguf` | spawn / terminate `llama-server` | 8080 |

All three runners are used by default. Pass `--runner` to restrict to a subset.

## Quick Start

### Run all runners against a corpus

```bash
pixi run python benchmark_suite.py --corpus http_server
```

### Run only llama.cpp

```bash
pixi run python benchmark_suite.py --corpus http_server --runner llama.cpp
```

### Run two runners

```bash
pixi run python benchmark_suite.py --corpus http_server --runner ollama,llama.cpp
```

### Filter by model size

Only benchmark models with ≥ 25 billion parameters:

```bash
pixi run python benchmark_suite.py --corpus http_server --min-size 25B
```

### Filter by keyword

Only benchmark models whose name (or path) contains `qwen` (case-insensitive):

```bash
pixi run python benchmark_suite.py --corpus http_server --includes qwen
```

Exclude models containing `0.5b` or `1.5b`:

```bash
pixi run python benchmark_suite.py --corpus http_server --excludes 0.5b,1.5b
```

For llama.cpp, `--includes` and `--excludes` match against the **full path** under `~/.lmstudio/models`, so parent directory names are searchable too:

```bash
# Matches models inside any directory that contains "MTP" in its name
pixi run python benchmark_suite.py --corpus http_server --runner llama.cpp --includes MTP
```

### Dry run

Preview what would run without loading models or executing benchmarks:

```bash
pixi run python benchmark_suite.py --corpus http_server --dry-run
```

### Clean run

Delete the previous results table and JSON dumps for the corpus, then run fresh:

```bash
pixi run python benchmark_suite.py --corpus http_server --clean-run
```

### Recreate the table without re-running

Rebuild the Markdown and HTML tables from existing JSON dumps in `results/`:

```bash
pixi run python benchmark_suite.py --corpus http_server --recreate-table
```

## CLI Reference

| Argument                | Required | Default                       | Description |
|-------------------------|----------|-------------------------------|-------------|
| `--corpus`              | **Yes**  | —                             | Corpus config name (e.g. `http_server`) or path to a source file. |
| `--runner` / `--runners`| No       | `ollama,lmstudio,llama.cpp`   | Comma-delimited runner(s). Choices: `ollama`, `lmstudio`, `llama.cpp`. |
| `--min-size`            | No       | —                             | Skip models smaller than N billion parameters. Accepts `25B` or `25`. |
| `--includes`            | No       | —                             | Comma-delimited keywords (case-insensitive). Only models whose name contains ANY keyword are benchmarked. For llama.cpp, the full path is searched. |
| `--excludes`            | No       | —                             | Comma-delimited keywords (case-insensitive). Models whose name contains ANY keyword are skipped. For llama.cpp, the full path is searched. |
| `--dry-run`             | No       | `False`                       | Print planned actions without loading models or running benchmarks. |
| `--clean-run`           | No       | `False`                       | Delete the existing results table and corpus JSON dumps before starting. |
| `--recreate-table`      | No       | `False`                       | Skip benchmarks; rebuild tables from existing JSON dumps in `results/`. |
| `--cooldown-seconds`    | No       | `30`                          | Seconds to wait between consecutive model benchmarks. |
| `--timeout`             | No       | `10000`                       | Maximum seconds allowed for a single model's benchmark subprocess. |
| `--load-verify-timeout` | No       | `180`                         | Seconds to poll a model's API endpoint after loading before declaring it unready. Set to `0` to skip. |
| `--save-table`          | No       | `results_table.md`            | Path for the Markdown results table. An HTML file is written alongside it. |
| `--corpora-dir`         | No       | `configs/corpora`             | Directory containing corpus TOML configs. |
| `--user-models-dir`     | No       | `configs/models/user`         | Directory where per-run model TOML configs are written. |
| `--template-dir`        | No       | `configs/models`              | Directory containing model template TOMLs. |

## Available Corpora

Corpus configs live in `configs/corpora/`:

| Name              | Description                          |
|-------------------|--------------------------------------|
| `http_server`     | HTTP server source code corpus       |
| `jquery`          | jQuery library source code           |
| `llm`             | LLM-related source code              |
| `python`          | Python standard library source code  |
| `vspaero_adjoint` | VSPAERO adjoint source code          |

You can also pass a direct file path instead of a corpus name:

```bash
pixi run python benchmark_suite.py --corpus /path/to/myfile.py
```

## Output

After a run completes:

| Path | Description |
|------|-------------|
| `results_table.md` | Markdown table with per-model results |
| `results_table.html` | Interactive HTML table — sortable columns (click header), resizable columns (drag border), per-column visibility toggles |
| `results/` | Per-model JSON dumps named `{corpus}__{model}.json` |
| `analysis/charts/` | Interactive HTML dashboards (if visualization succeeded) |

## How Runners Are Skipped

If a runner has no models installed (or all models are filtered out by `--min-size`, `--includes`, or `--excludes`), that runner is silently skipped with a stderr message and the next runner proceeds:

```
No .gguf models found under /Users/you/.lmstudio/models; skipping llama.cpp.
No models meet the --min-size 25B threshold; skipping Ollama.
```

## Architecture

```
benchmark_suite.py
├── Step 1:   Discover models
│             ├── lmstudio  →  lms ls
│             ├── ollama    →  ollama ls
│             └── llama.cpp →  scan ~/.lmstudio/models/**/*.gguf
├── Step 2:   Validate corpus
├── Step 3:   Generate per-model TOML configs
├── Step 3.5: Unload all currently loaded models
├── Step 4:   Run benchmarks
│             └── for each model: load → verify ready → bench.py run → unload
├── Step 5:   Parse results from JSON dumps
├── Step 6:   Generate Markdown + HTML results tables
└── Step 7:   Run visualization (analysis/visualize.py)
```

After all runners finish, a **consolidated table** is generated combining results across all runners.
