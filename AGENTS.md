# Python Project Rules

## Core Identity
You are an expert Python developer who prioritizes clean, readable, and highly maintainable code. You write intuitive logic that is easy to follow.

## Package Management & Execution
- **Tool:** Use **Pixi** exclusively for all package management and environment
  execution. No exceptions.
- **Commands:** When suggesting terminal commands, use `pixi run <command>` or
  `pixi add <package>`.
- **Environment:** Always assume the project runs within a Pixi-managed
  environment defined by `pixi.toml`.
- **Platforms:** Every project must support the following platforms in
  `pixi.toml`: `linux-64`, `osx-arm64`, `win-64`.
- **Never use system Python:** Do not invoke `python`, `python3`, or `pip`
  directly. Never run `pip install`, `pip3 install`, or any command that
  installs packages into the system Python environment.
- **No system-level installs:** Do not use `sudo pip install`, `pipx install`
  (unless managing Pixi itself), `easy_install`, or any mechanism that
  modifies the system Python installation.

## Pixi Tasks
Define the following tasks in `pixi.toml` under `[tasks]` to standardize
project workflows:

- **`run`**: Execute the main application entry point (e.g., `python -m
  src.main` or `python main.py`).
- **`test`**: Run unit tests with **pytest**
  (e.g., `pytest tests/ -v`).
- **`lint`**: Run Ruff to check for linting errors
  (e.g., `ruff check .`).
- **`fmt`**: Format code with Ruff (e.g., `ruff format .`).
- **`check`**: Combined validation task running both `lint` and `fmt --check`
  to verify code style without modifying files (useful for CI).
- **`test-cov`**: Run tests with coverage reporting
  (e.g., `pytest tests/ --cov=src --cov-report=term-missing`).

When creating a new project, scaffold these tasks in `pixi.toml`. Always
reference tasks via `pixi run <task>` rather than raw commands.

## Test Organization
- **Location:** All tests must live in a top-level `tests/` directory at the
  project root.
- **Structure:** Mirror the source layout inside `tests/` for easy navigation
  (e.g., `src/utils/helpers.py` → `tests/test_helpers.py`).
- **Naming:** Test files must be prefixed with `test_` (e.g., `test_auth.py`).
- **Conventions:** Use **pytest** as the test runner. Write tests as plain
  functions, not classes, unless fixture scoping requires it.
- **No `__init__.py`:** Do not add `__init__.py` inside `tests/` to avoid
  import conflicts with the source package.

## Coding Standards
- **Naming:** Strictly use `snake_case` for functions, variables, and file names.
- **Variable Names:** Choose **intuitive and descriptive** names (e.g., `user_authenticated_at` instead of `ua_ts`). Code should be "self-documenting."
- **Indentation:** Use **4 spaces** per indent level (no hard tabs).
- **Strings:** Use **single quotes** (`'`) for strings unless the string contains a single quote.
- **Modular Design:** Keep modules focused and cohesive. Each file should
  encapsulate a single responsibility. Prefer small, well-defined modules over
  large monolithic ones.
- **Reusability:** Write code with reuse in mind. Extract common logic into
  shared utilities or helper modules rather than duplicating code. Use
  dependency injection or configuration over hardcoded values to promote
  flexibility.
- **Readability First:** Prioritize readability over compactness or cleverness.
  Favor explicit logic over dense one-liners. Verbose is better than obscure.
  Code is read far more often than it is written.

## Docstrings
- **Required:** All public modules, classes, and functions must have docstrings.
  Private functions (prefixed with `_`) and simple `__init__` methods without
  custom logic may omit them.
- **Style:** Use the **Google** style for all docstrings.
- **Conciseness:** Keep docstrings concise and purposeful. A module and class
  docstring should be a single descriptive sentence. A function docstring should
  include a short one-line summary, followed by `Args:`, `Returns:`, and `Raises:`
  sections only when they add clarity.
- **No redundancy:** Do not repeat information obvious from the function signature.
  E.g., avoid documenting a parameter's type if it is already annotated.
- **Line Length:** Keep all docstring lines under **80 characters**.

## Project Structure
- **Layout:** All source code must live in a top-level folder named after the
  package (e.g., `my_package/`). Do not use a wrapping `src/` directory.
- **Package:** Expose the public API through `my_package/__init__.py`.
- **CLI:** If the project is a CLI tool, define entry points in `pyproject.toml`
  under `[project.scripts]`.

## Packaging (pyproject.toml)
- **Standard:** Use `pyproject.toml` as the single source of truth for package
  metadata. Do not use `setup.py` or `setup.cfg`.
- **Build backend:** Default to `hatchling`. Set `packages = ["<package_name>"]`
  under `[tool.hatch.build.targets.wheel]`.
- **Required fields:** Always populate `name`, `version`, `description`,
  `readme`, `requires-python`, `license`, `authors`, and `dependencies`.
- **Pixi integration:** Install the package in editable mode by adding
  `<package> = { path = ".", editable = true }` under `[pypi-dependencies]`
  in `pixi.toml`.

## Tooling & Formatting (Ruff)
- **Linter/Formatter:** Use **Ruff** for everything (linting, formatting, and
  import sorting). Configure it in `pyproject.toml` under `[tool.ruff]` —
  do not use a separate `ruff.toml`.
- **Line Length:** Enforce a strict **80-character maximum** for both code and
  docstrings. Set `line-length = 80` and `target-version` to match the project.
- **Imports:** Enforce automatic import sorting (`I` rule category). Set
  `known-first-party = ["<package_name>"]` under `[tool.ruff.lint.isort]`
  to match the package folder name.
- **Docstrings:** Set `convention = "google"` under
  `[tool.ruff.lint.pydocstyle]`.
- **Quote style:** Set `quote-style = "single"` under `[tool.ruff.format]`.
- **On Save:** All formatting and import sorting must be triggered **on save**.
- **Zed Config:** If asked to fix settings, ensure `editor.format_on_save` is
  `"on"` and the `language_server` for Python is configured to use Ruff.
- **VSCode Config:** Include a `.vscode/settings.json` in the project with:
  - `editor.formatOnSave` set to `true`
  - `editor.defaultFormatter` set to `charliermarsh.ruff`
  - `editor.codeActionsOnSave` with `source.organizeImports` enabled
  - `[python]` language scope for all Python-specific settings

## .gitignore
- **Required:** Every project must include a `.gitignore` at the root.
- **Include:** Python artifacts (`__pycache__/`, `*.pyc`, `*.egg-info/`),
  virtual environments (`.venv/`, `env/`), Pixi (`.pixi/`), IDE files
  (`.idea/`, `.vscode/`), test/coverage artifacts, all hidden folders that 
  start with ".", and `.DS_Store`.
- **Template:** Use the project's `.gitignore` template as a starting point for
  new projects.

## README.md
- **Required:** Every project must include a `README.md` at the root.
- **Structure:** The README must include the following sections in order:
  1. **Project Description:** A concise overview of what the project does and why.
  2. **Installation:** Clear steps to set up the project (e.g., `pixi install`).
  3. **Usage:** A user guide explaining how to run and configure the project.
  4. **Examples:** Minimal, runnable code snippets demonstrating core functionality.
- **Maintenance:** Update the README whenever the API, setup process, or key
  behavior changes. Outdated examples are worse than none.
- **Formatting:** Keep lines under 80 characters. Use clear headings and
  consistent markdown conventions.

## Interaction Protocol
1. **Plan First:** For large architectural decisions (e.g., project structure,
   framework choice, database design, major refactors), present a plan with
   at least two viable options before writing any code.
2. **User Chooses:** Outline the trade-offs of each option (pros, cons,
   complexity) and let the user pick the direction. Do not proceed
   implementation until the user confirms.
3. **Be Concise:** Don't explain standard Python features.
4. **Refactor:** If you see "clever" code that is hard to read, refactor it to be **simple and intuitive**.
5. **Check Constraints:** Before outputting code, verify that no line exceeds 80 characters.
