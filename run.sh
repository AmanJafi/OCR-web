#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Starting Optical Character Recognition Web App..."
if [ -f "./venv/bin/python" ]; then
    ./venv/bin/python app.py
else
    python3 app.py
fi
