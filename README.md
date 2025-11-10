# cell2sentence4longevity

Repository for longevity2cells4longevity finetuning and supporting code.

## Setup

This project uses `uv` for dependency management. To set up the project:

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install project dependencies:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

Or use `uv run` to run commands directly without activating the environment:
```bash
uv run python your_script.py
```

