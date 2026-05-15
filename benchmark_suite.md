# Benchmark Suite

Benchmark suite orchestrator for benchmarking LLM models running under **LMStudio** and **Ollama**. It automates the full pipeline: discovering installed models, generating per-model config files, running sequential benchmarks, and producing results tables with visual reports.

## What It Does

1. **Discovers models** via `lms ls` (LMStudio) or `ollama ls` (Ollama).
2. **Generates per-model TOML configs** from templates in `configs/models/`, injecting the correct `base_url` for the selected runner.
3. **Runs benchmarks** sequentially — loading each model, running `bench.py run`, then unloading it.
4. **Produces a Markdown results table** (with per-model pass rate, hallucinated lines, bonus matched lines, and runtime).
5. **Generates an HTML results table** with sortable columns.
6. **Runs a visualization script** (`analysis/visualize.py`) to produce interactive HTML dashboards.

## Prerequisites

- [pixi](https://pixi.sh/) installed
- Either **LMStudio** (`lms` CLI) or **Ollama** installed and running
- Corpus TOMLs in `configs/corpora/` or a single source file path

## Quick Start

### Benchmark a single corpus with LMStudio (default)

```bash
pixi run python benchmark_suite.py --corpus http_server
```

### Benchmark with Ollama

```bash
pixi run python benchmark_suite.py --corpus http_server --runner ollama
```

### Benchmark with both runners (comma-delimited)

```bash
pixi run python benchmark_suite.py --corpus http_server --runner lmstudio,ollama
```

### Filter by model size

Only benchmark models with >=25 billion parameters:

```bash
pixi run python benchmark_suite.py --corpus jquery --min-size 25B
```

### Dry run (no benchmarks executed)

```bash
pixi run python benchmark_suite.py --corpus jquery --dry-run
```

### Clean run (delete previous results first)

```bash
pixi run python benchmark_suite.py --corpus http_server --clean-run
```

### Filter by keyword

Only benchmark models whose names contain `qwen`:

```bash
pixi run python benchmark_suite.py --corpus jquery --includes qwen
```

Exclude models containing `3b`:

```bash
pixi run python benchmark_suite.py --corpus jquery --excludes 3b
```

## CLI Reference

| Argument              | Required | Default                        | Description                                                                                   |
|-----------------------|----------|--------------------------------|-----------------------------------------------------------------------------------------------|
| `--corpus`            | Yes      | —                              | Corpus config name (e.g. `http_server`) or path to a single source file.                     |
| `--runner`            | No       | `lmstudio`                     | Model runner(s), comma-delimited. Choices: `lmstudio`, `ollama`.                              |
| `--min-size`          | No       | —                              | Only benchmark models with >= this many parameters (in billions). Accepts `25B` or `25`.      |
| `--corpora-dir`       | No       | `configs/corpora`              | Directory containing corpus TOML files.                                                       |
| `--user-models-dir`   | No       | `configs/models/user`          | Directory to write per-model config TOMLs.                                                    |
| `--template-dir`      | No       | `configs/models`               | Directory containing template TOML files.                                                     |
| `--dry-run`           | No       | `False`                        | Print what would happen without executing benchmarks.                                         |
| `--clean-run`         | No       | `False`                        | Delete existing results tables and corpus JSON dumps before running.                          |
| `--save-table`        | No       | `results_table.md`             | Path for the Markdown results table.                                                          |
| `--cooldown-seconds`  | No       | 10                             | Seconds to wait between per-model benchmarks.                                                 |
| `--timeout`           | No       | 3600                           | Per-model subprocess timeout in seconds.                                                      |
| `--includes`          | No       | —                              | Comma-delimited keywords; only models containing ANY keyword are benchmarked (case-insensitive).|
| `--excludes`          | No       | —                              | Comma-delimited keywords; models containing ANY keyword are excluded (case-insensitive).      |

## Available Corpora

Corpus configs live in `configs/corpora/`:

| Name                | Description                         |
|---------------------|-------------------------------------|
| `http_server`       | HTTP server source code corpus      |
| `jquery`            | jQuery library source code          |
| `llm`               | LLM-related source code             |
| `python`            | Python standard library source code |
| `vspaero_adjoint`   | VSPAERO adjoint source code         |

You can also pass a direct file path instead of a corpus name:

```bash
pixi run python benchmark_suite.py --corpus /path/to/myfile.py
```

## Output

After a run completes, you'll find:

- **`results_table.md`** — Markdown table with per-model results
- **`results_table.html`** — Sortable HTML table
- **`results/`** — Per-model JSON dumps
- **`analysis/charts/`** — Interactive HTML dashboards (if visualization succeeded)

## Skipping Runners

If a runner has no models installed (e.g., `ollama ls` returns nothing), that runner is silently skipped with a stderr message:

```
No models found via 'ollama ls'; skipping Ollama.
```

Similarly, if `--min-size`, `--includes`, or `--excludes` filters eliminate all models for a given runner, that runner is skipped.

## Architecture

```
benchmark_suite.py
├── Step 1: Discover models (lms ls / ollama ls)
├── Step 2: Validate corpus
├── Step 3: Generate per-model TOML configs
├── Step 3.5: Unload all loaded models
├── Step 4: Run benchmarks (per-model load -> bench.py run -> unload)
├── Step 5: Parse results from JSON dumps
├── Step 6: Generate Markdown + HTML tables
└── Step 7: Run visualization (analysis/visualize.py)
```
