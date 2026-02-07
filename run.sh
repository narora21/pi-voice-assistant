#!/usr/bin/env bash

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "Starting Pi Voice Assistant..."

# -----------------------------
# 1. Ensure uv is installed
# -----------------------------
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# -----------------------------
# 2. Ensure correct Python
# -----------------------------
echo "Ensuring Python 3.11..."
uv python install 3.11

# -----------------------------
# 3. Create venv if missing
# -----------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

# -----------------------------
# 4. Install dependencies
# -----------------------------
echo "Syncing dependencies..."
uv sync

# -----------------------------
# 5. Run application
# -----------------------------
echo "Launching assistant..."
uv run -m src.main
