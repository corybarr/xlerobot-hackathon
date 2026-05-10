#!/usr/bin/env bash
# Deploy a bore.pub tunnel from Spark to expose the gemma-proxy publicly,
# without sudo and without granting ssh access to the team.
#
# bore is a single user-space TCP tunnel binary (no signup, no daemon needed).
# Maps localhost:18080 on Spark to bore.pub:<random_port>. Anyone with the
# bearer token can reach the proxy via that public URL.
#
# Usage:
#   ./scripts/deploy_bore_tunnel.sh
#
# Env:
#   SPARK_HOST   default dgx-spark
#   PROXY_PORT   default 18080  (the local port bore forwards)

set -euo pipefail

SPARK_HOST="${SPARK_HOST:-dgx-spark}"
PROXY_PORT="${PROXY_PORT:-18080}"

echo "==> Installing bore on Spark (user-space, ~/bin/) if missing"
ssh "${SPARK_HOST}" 'set -e
  mkdir -p ~/bin ~/gemma-proxy
  if [ ! -x ~/bin/bore ]; then
    ARCH=$(uname -m)
    case $ARCH in
      aarch64|arm64) BORE_ASSET="aarch64-unknown-linux-musl" ;;
      x86_64)        BORE_ASSET="x86_64-unknown-linux-musl" ;;
      *) echo "  unsupported arch $ARCH"; exit 1 ;;
    esac
    echo "  arch=$ARCH  asset=$BORE_ASSET"
    BORE_URL=$(curl -sL https://api.github.com/repos/ekzhang/bore/releases/latest \
      | grep browser_download_url \
      | grep "${BORE_ASSET}.tar.gz" \
      | head -1 | cut -d\" -f4)
    if [ -z "$BORE_URL" ]; then echo "  could not find bore release for $BORE_ASSET"; exit 1; fi
    echo "  downloading $(basename $BORE_URL)..."
    curl -sL "$BORE_URL" -o /tmp/bore.tar.gz
    tar -xzf /tmp/bore.tar.gz -C /tmp
    mv /tmp/bore ~/bin/bore
    chmod +x ~/bin/bore
    rm -f /tmp/bore.tar.gz
  fi
  echo "  bore: $(~/bin/bore --version)"
'

echo "==> Stopping any existing bore tunnel (via pidfile)"
ssh "${SPARK_HOST}" '
  PF=~/gemma-proxy/bore.pid
  if [ -f "$PF" ]; then
    PID=$(cat "$PF" 2>/dev/null || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null || true
      sleep 0.5
    fi
    rm -f "$PF"
  fi
  rm -f ~/gemma-proxy/bore.log
  echo "  done"
'

echo "==> Starting bore tunnel (localhost:${PROXY_PORT} -> bore.pub:RANDOM)"
ssh "${SPARK_HOST}" "nohup ~/bin/bore local ${PROXY_PORT} --to bore.pub > ~/gemma-proxy/bore.log 2>&1 & echo \$! > ~/gemma-proxy/bore.pid"

echo "  waiting for assigned port..."
PUBLIC_PORT=""
for _ in $(seq 1 20); do
  sleep 1
  # bore prints ANSI color codes around remote_port=, so we strip them first.
  PUBLIC_PORT="$(ssh "${SPARK_HOST}" "sed -E 's/\\x1b\\[[0-9;]*m//g' ~/gemma-proxy/bore.log 2>/dev/null | grep -oE 'bore\\.pub:[0-9]+' | tail -1 | cut -d: -f2" || true)"
  [ -n "${PUBLIC_PORT}" ] && break
done

if [ -z "${PUBLIC_PORT}" ]; then
  echo "  ERROR: bore did not announce a port within 20s. Last 15 log lines:" >&2
  ssh "${SPARK_HOST}" "tail -15 ~/gemma-proxy/bore.log" >&2
  exit 2
fi

PUBLIC_URL="http://bore.pub:${PUBLIC_PORT}"
echo "  public-port: ${PUBLIC_PORT}"

echo "==> Smoke test from the local tablet"
TOKEN_FROM_PROXY="$(ssh "${SPARK_HOST}" "tr '\\0' '\\n' < /proc/\$(cat ~/gemma-proxy/proxy.pid)/environ 2>/dev/null | grep ^PROXY_TOKEN= | cut -d= -f2-")"
if [ -z "${TOKEN_FROM_PROXY}" ]; then
  echo "  WARN: could not recover token from proxy process env. Use the token from your last deploy_gemma_proxy.sh run." >&2
else
  RESP="$(curl -s --max-time 10 -H "Authorization: Bearer ${TOKEN_FROM_PROXY}" "${PUBLIC_URL}/api/tags" || true)"
  if echo "${RESP}" | grep -q gemma3:27b; then
    echo "  smoke: ok — ${PUBLIC_URL} responds with gemma3:27b"
  else
    echo "  smoke: WARN — got: ${RESP:0:200}" >&2
  fi
fi

cat <<EOF

bore tunnel live (no ssh handout, no sudo, public URL)

  PUBLIC url:  ${PUBLIC_URL}
  forwards to: localhost:${PROXY_PORT} on ${SPARK_HOST} (the gemma-proxy)
  token:       (use the existing token from deploy_gemma_proxy.sh — bore doesn't
                touch auth, the proxy still enforces it)

Caveats:
  - http only (no TLS). For demo Railway frontend, wrap with HTTPS server-side.
  - port changes if bore restarts. To stop: ssh ${SPARK_HOST} 'kill \$(cat ~/gemma-proxy/bore.pid)'
  - bore.pub is community-run free service. Reasonable for hackathon, not prod.

EOF
