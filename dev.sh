#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up paths
VENV_DIR="$HOME/.local/share/manga_ocr/pyenv"
PYTHON_PATH="$VENV_DIR/bin/python"

# Create virtualenv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Debug info
if [[ "$*" == *"-d"* ]] || [[ "$*" == *"--debug"* ]]; then
    echo "[DEBUG] Using Python: $PYTHON_PATH"
    echo "[DEBUG] Python version: $($PYTHON_PATH --version)"
    echo "[DEBUG] Working directory: $SCRIPT_DIR"
    echo "[DEBUG] Virtual environment: $VENV_DIR"
fi

# Execute the Python script with all passed arguments
exec "$PYTHON_PATH" "$SCRIPT_DIR/src/transformers_ocr.py" "$@" 