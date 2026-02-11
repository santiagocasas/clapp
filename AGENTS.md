# AGENTS

This file is for agentic coding assistants working in this repo.
Keep guidance concise, actionable, and aligned with the existing codebase.

## Repo Snapshot
- Project: CLAPP (Streamlit app for CLASS-focused LLM assistance)
- Language: Python 3.11+ (see `pyproject.toml`)
- Package layout: `clapp/` package, plus scripts and data directories
- Dependency manager: `uv` (recommended)
- Lint/format: Ruff (configured in `pyproject.toml`)

## Cursor/Copilot Rules
- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` found.
- If added later, follow those instructions and update this file.

## Build / Run / Lint / Test

### Setup
- `uv venv .venv --python 3.11`
- `source .venv/bin/activate`
- `uv sync`
- Dev tools: `uv sync --extra dev`

### Run the app
- `uv run streamlit run clapp/app.py`

### Build (packaging)
- Recommended: `uv build` (uses Hatchling via `pyproject.toml`)
- Alternative: `python -m build` (requires `build` installed)

### Lint
- `uv run ruff check .`
- Auto-fix: `uv run ruff check . --fix`

### Format
- `uv run ruff format .`

### Tests
- There is no pytest suite in this repo. Tests are script-based.
- CLASS smoke test: `uv run python test_classy.py`
- Blablador API manual test: `uv run python minimal_blablador_test.py`

### Single-test guidance
- Run any standalone script directly, for example:
  - `uv run python test_classy.py`
  - `uv run python scripts/refresh_class_data.py --help`

## Code Style Guidelines

### Formatting
- Use Ruff format (`ruff format`) and keep changes compatible with it.
- Line length target: 88 chars (Ruff), but E501 is ignored.
- Use double quotes for strings unless a single-quote avoids escaping.
- Indentation: 4 spaces, LF line endings.

### Ruff Configuration Notes
- Lint selects: E, F, I (errors, pyflakes, import sorting).
- Per-file ignore: `clapp/app.py` ignores E402 (import position).
- Exclusions include: `.clappenv`, `.venv`, `class-data`, `indexes`, `plots`.
- `clapp_legacy.py` is excluded from Ruff checks.

### Imports
- Follow Ruff "I" (isort) ordering: standard library, third-party, local.
- Prefer absolute imports from the `clapp` package.
- Avoid wildcard imports; keep import lists explicit.
- If you add conditional imports, follow the `langchain_compat.py` pattern.

### Types
- Use Python 3.11 type syntax (`str | None`, `list[str]`, `dict[str, Any]`).
- Add type hints for public functions or non-trivial data flows.
- Use `Annotated` only when metadata is required (see `groupchat.py`).

### Naming
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants and module-level settings: `UPPER_SNAKE_CASE`.
- Avoid single-letter names outside short loops or math contexts.

### Error Handling
- Keep the Streamlit UI responsive; prefer graceful fallbacks.
- In UI code, surface errors with `st.error` and use `st.exception` for detail.
- For recoverable failures, return safe defaults (empty lists/strings).
- Centralize LLM error messages via `clapp.utils.llm_errors.format_llm_error`.

### State Management (Streamlit)
- Use `st.session_state` for persistent UI state; guard missing keys.
- Initialize session keys early (see `clapp/app.py` and `init_session`).
- Keep chat history in `st.session_state.messages` (role/content dicts).

### Streamlit UI Patterns
- Use `st.chat_message` for chat transcript and `st.chat_input` for user input.
- Prefer `st.markdown` for rich text and controlled HTML blocks.
- After mutating state, call `st.rerun()` when the UI must refresh.
- Use `st.columns`, `st.toggle`, and `st.pills` to match existing layout.

### LLM and RAG Conventions
- Build models via `clapp.llms.providers.build_llm`.
- Use `clapp.llms.providers.build_embeddings` for embeddings.
- Normalize base URLs using `clapp.config.normalize_base_url`.
- Fetch OpenAI base URLs with `clapp.config.get_openai_base_url`.
- Retrieval context helpers live in `clapp.rag.pipeline`.

### Agents and Orchestration
- Agent configs and review/refine flow live in `clapp/agents/groupchat.py`.
- Primary response orchestration is in `clapp/services/orchestrator.py`.
- Keep the fast-mode streaming path lightweight and resilient.
- Store debug traces in `st.session_state.debug_messages` when enabled.

### Code Execution
- Code execution utilities live in `clapp/execution/code_execution.py`.
- Preserve the single-code-block contract for executable responses.
- Only adjust execution flows if you also update safety checks and parsing.

### File and Directory Notes
- `class-data/` is source material for RAG. Do not edit manually.
  - To refresh it, use `scripts/refresh_class_data.py`.
- `plots/` contains generated artifacts and scripts; avoid manual edits.
- `prompts/` contains agent instructions; edits change model behavior.

### Assets and Generated Files
- `plots/*.py` and `plots/*.png` are generated outputs (git-ignored).
- `class-data/` content should be treated as read-only input.
- Avoid formatting or linting large data assets and cached outputs.

### Security and Secrets
- Never log or print API keys or secrets.
- Use `clapp.config.get_local_secret` for local secrets.
- Local secrets live in `.streamlit/secrets.toml` (git-ignored).

### General Practices
- Preserve existing patterns in Streamlit UI (chat messages, sidebar state).
- Prefer small, focused functions and predictable return types.
- Keep new modules within the `clapp/` package unless they are scripts.
- Avoid adding new dependencies unless required and aligned with project scope.
- Add new dependencies to `pyproject.toml` and keep versions pinned.

## Useful Paths
- App entry: `clapp/app.py`
- LLM providers: `clapp/llms/providers.py`
- RAG pipeline: `clapp/rag/pipeline.py`
- Code execution: `clapp/execution/code_execution.py`
- CLASS validation: `clapp/domain/classy/validation.py`
- Data refresh script: `scripts/refresh_class_data.py`
