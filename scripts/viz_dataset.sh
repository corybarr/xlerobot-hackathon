#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
lerobot-dataset-viz --repo-id "${1:-lerobot/pusht}" --episode-index "${2:-0}"
