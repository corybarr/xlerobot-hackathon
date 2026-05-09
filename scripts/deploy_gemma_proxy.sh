#!/usr/bin/env bash
# Deploy the gemma_proxy to Spark and start it under nohup.
#
# Usage:
#   ./scripts/deploy_gemma_proxy.sh                  # generates a fresh token
#   PROXY_TOKEN=<existing> ./scripts/deploy_gemma_proxy.sh   # reuse a known token
#
# Env:
#   SPARK_HOST    default dgx-spark
#   PROXY_PORT    default 18080
#   ALLOWED_MODEL default gemma3:27b
#
# Side effects on Spark (none outside ~/gemma-proxy/):
#   - mkdir ~/gemma-proxy/
#   - scp gemma_proxy.py and proxy.env into it
#   - kill any existing gemma-proxy process belonging to your user
#   - nohup-spawn a new one
#
# Prints the URL + token + ready-to-paste env block on success.

set -euo pipefail

SPARK_HOST="${SPARK_HOST:-dgx-spark}"
PROXY_PORT="${PROXY_PORT:-18080}"
ALLOWED_MODEL="${ALLOWED_MODEL:-gemma3:27b}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_FILE="${REPO_ROOT}/scripts/spark/gemma_proxy.py"

if [[ ! -f "${PROXY_FILE}" ]]; then
  echo "ERROR: ${PROXY_FILE} not found" >&2
  exit 1
fi

# Generate token if not provided. openssl is on every Linux box; fall back to /dev/urandom.
if [[ -z "${PROXY_TOKEN:-}" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    PROXY_TOKEN="$(openssl rand -hex 32)"
  else
    PROXY_TOKEN="$(head -c 32 /dev/urandom | xxd -p -c 64)"
  fi
fi

echo "==> Verifying Spark reachable"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${SPARK_HOST}" 'echo "  spark up: $(hostname) (user: $USER)"'

echo "==> Verifying Ollama is up on Spark"
ssh "${SPARK_HOST}" "curl -sf http://127.0.0.1:11434/api/tags >/dev/null && echo '  ollama: ok' || (echo 'ollama not reachable' >&2; exit 2)"

echo "==> Verifying model '${ALLOWED_MODEL}' is pulled"
ssh "${SPARK_HOST}" "ollama list | awk 'NR>1{print \$1}' | grep -Fxq '${ALLOWED_MODEL}' && echo '  model present' || (echo 'model ${ALLOWED_MODEL} not pulled. run: ssh ${SPARK_HOST} \"ollama pull ${ALLOWED_MODEL}\"' >&2; exit 3)"

echo "==> Deploying proxy code"
ssh "${SPARK_HOST}" "mkdir -p ~/gemma-proxy"
scp -q "${PROXY_FILE}" "${SPARK_HOST}:~/gemma-proxy/gemma_proxy.py"

echo "==> Stopping any existing proxy (via pidfile, avoids pkill self-match)"
ssh "${SPARK_HOST}" '
  PF=~/gemma-proxy/proxy.pid
  if [ -f "$PF" ]; then
    PID=$(cat "$PF" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
      sleep 0.5
      kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$PF"
  fi
  echo "  done"
'

echo "==> Starting proxy (nohup, logs to ~/gemma-proxy/proxy.log)"
ssh "${SPARK_HOST}" "PROXY_TOKEN='${PROXY_TOKEN}' PROXY_PORT='${PROXY_PORT}' ALLOWED_MODEL='${ALLOWED_MODEL}' nohup python3 ~/gemma-proxy/gemma_proxy.py > ~/gemma-proxy/proxy.log 2>&1 & echo \$! > ~/gemma-proxy/proxy.pid"
sleep 1.5

echo "==> Smoke test (from Spark itself — proxy is on the LAN-side IP)"
SPARK_IP="$(ssh "${SPARK_HOST}" "ip -4 addr show | awk '/inet / && !/127.0/ {print \$2}' | cut -d/ -f1 | head -1")"
URL="http://${SPARK_IP}:${PROXY_PORT}"
SMOKE_RESP="$(ssh "${SPARK_HOST}" "curl -s --max-time 5 -H 'Authorization: Bearer ${PROXY_TOKEN}' http://localhost:${PROXY_PORT}/api/tags" || true)"
SMOKE_OK=0
if echo "${SMOKE_RESP}" | grep -q "${ALLOWED_MODEL}"; then
  SMOKE_OK=1
  echo "  smoke: ok — proxy returned ${ALLOWED_MODEL}"
else
  echo "  smoke: FAILED. response:" >&2
  echo "    ${SMOKE_RESP}" >&2
  echo "  last 20 lines of proxy.log:" >&2
  ssh "${SPARK_HOST}" "tail -20 ~/gemma-proxy/proxy.log 2>&1 || true" >&2
fi

# Always print the token — even if the smoke test failed, the token is still
# valid and you may need it to debug or re-test manually.
cat <<EOF

gemma-proxy on Spark
  url:      ${URL}      (LAN — direct from same-network team members)
  also at:  http://localhost:${PROXY_PORT}      (after: ssh -fNL ${PROXY_PORT}:localhost:${PROXY_PORT} ${SPARK_HOST})
  model:    ${ALLOWED_MODEL}
  token:    ${PROXY_TOKEN}

Use from a client (orchestrator/watcher/ask all read these env vars):
  export OLLAMA_HOST="${URL}"            # or http://localhost:${PROXY_PORT} via tunnel
  export GEMMA_PROXY_TOKEN="${PROXY_TOKEN}"

Rotate the token:  ./scripts/rotate_gemma_token.sh
Stop the proxy:    ssh ${SPARK_HOST} 'kill \$(cat ~/gemma-proxy/proxy.pid)'

EOF

[ "${SMOKE_OK}" = "1" ] || exit 4
