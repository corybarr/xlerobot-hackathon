#!/usr/bin/env bash
# Deploy Gemma on Spark (Ollama) and start the orchestrator on the local tablet.
#
# Usage: ./scripts/deploy_orchestrator.sh
# Env:   GEMMA_MODEL (default gemma3:27b), GOAL (default "set the table")
#
# Requires:
#   - Spark reachable via 'ssh dgx-spark' (run spark-check first)
#   - Ollama installed on Spark
#   - python deps: requests, opencv-python, pyyaml (lerobot env covers all 3)

set -euo pipefail

GEMMA_MODEL="${GEMMA_MODEL:-gemma3:27b}"
GOAL="${GOAL:-set the table}"

echo "==> Spark check"
spark-check >/dev/null 2>&1 || { echo "Spark unreachable. Aborting."; exit 1; }
echo "    OK"

echo "==> Pulling ${GEMMA_MODEL} on Spark (idempotent)"
ssh dgx-spark "ollama pull ${GEMMA_MODEL}"

echo "==> Ensuring Ollama serve is running on Spark"
ssh dgx-spark "pgrep -f 'ollama serve' >/dev/null || nohup ollama serve > /tmp/ollama.log 2>&1 &"

echo "==> Forwarding Spark Ollama port (11434) to local tablet"
# Kill any stale tunnel first
pkill -f "ssh.*-L 11434:localhost:11434.*dgx-spark" 2>/dev/null || true
ssh -fNL 11434:localhost:11434 dgx-spark
sleep 1

echo "==> Sanity: GET /api/tags via tunnel"
curl -s --max-time 5 http://localhost:11434/api/tags | python -c "import json,sys; print('models:', [m['name'] for m in json.load(sys.stdin).get('models',[])])"

echo "==> Starting orchestrator (goal: '${GOAL}')"
GOAL="${GOAL}" GEMMA_MODEL="${GEMMA_MODEL}" python orchestrator/orchestrator.py
