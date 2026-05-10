#!/usr/bin/env bash
# Watchdog that keeps the gemma-proxy AND its bore tunnel alive.
# Designed to run on Spark via nohup (or systemd --user). Polls every 30s.
# Auto-restarts whichever component died.

set -u

OUT=~/gemma-proxy
PROXY_PIDFILE="${OUT}/proxy.pid"
BORE_PIDFILE="${OUT}/bore.pid"
LOG="${OUT}/watchdog.log"
PORT=18080

# Recover the proxy's env so a restart matches the original config.
# We snapshot the running proxy's env to disk on first watchdog start; later
# restarts read from that snapshot. If the snapshot is missing AND the proxy
# is currently running, take one now.
ENV_SNAPSHOT="${OUT}/proxy.env"

snapshot_proxy_env() {
    if [ -f "$PROXY_PIDFILE" ] && kill -0 "$(cat "$PROXY_PIDFILE")" 2>/dev/null; then
        local pid; pid="$(cat "$PROXY_PIDFILE")"
        # extract the env we care about (PROXY_TOKEN, PROXY_PORT, ALLOWED_MODEL, OLLAMA_URL)
        for k in PROXY_TOKEN PROXY_PORT ALLOWED_MODEL OLLAMA_URL PROXY_BIND; do
            v=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep "^${k}=" | head -1 | cut -d= -f2-)
            [ -n "$v" ] && printf 'export %s=%q\n' "$k" "$v" >> "${ENV_SNAPSHOT}.tmp"
        done
        if [ -s "${ENV_SNAPSHOT}.tmp" ]; then
            mv "${ENV_SNAPSHOT}.tmp" "$ENV_SNAPSHOT"
            chmod 600 "$ENV_SNAPSHOT"
        else
            rm -f "${ENV_SNAPSHOT}.tmp"
        fi
    fi
}

start_proxy() {
    if [ ! -f "$ENV_SNAPSHOT" ]; then
        echo "[$(date -Iseconds)] PROXY: env snapshot missing, cannot restart cleanly" | tee -a "$LOG"
        return 1
    fi
    # shellcheck disable=SC1090
    source "$ENV_SNAPSHOT"
    nohup python3 ~/gemma-proxy/gemma_proxy.py >> "${OUT}/proxy.log" 2>&1 &
    echo $! > "$PROXY_PIDFILE"
    echo "[$(date -Iseconds)] PROXY: restarted pid=$(cat "$PROXY_PIDFILE")" | tee -a "$LOG"
}

start_bore() {
    nohup ~/bin/bore local "$PORT" --to bore.pub >> "${OUT}/bore.log" 2>&1 &
    echo $! > "$BORE_PIDFILE"
    sleep 4
    PORT_ASSIGNED=$(sed -E 's/\x1b\[[0-9;]*m//g' "${OUT}/bore.log" | grep -oE 'bore\.pub:[0-9]+' | tail -1 | cut -d: -f2)
    echo "[$(date -Iseconds)] BORE: restarted pid=$(cat "$BORE_PIDFILE") public=bore.pub:${PORT_ASSIGNED:-?}" | tee -a "$LOG"
}

# One-time: snapshot env on first run if proxy is currently up
[ ! -f "$ENV_SNAPSHOT" ] && snapshot_proxy_env

echo "[$(date -Iseconds)] WATCHDOG: started pid=$$ poll=30s" | tee -a "$LOG"

while true; do
    # Check proxy
    if [ ! -f "$PROXY_PIDFILE" ] || ! kill -0 "$(cat "$PROXY_PIDFILE")" 2>/dev/null; then
        echo "[$(date -Iseconds)] PROXY: dead, restarting" | tee -a "$LOG"
        start_proxy
    fi

    # Check bore (only if proxy is up — pointless to bore an absent service)
    if [ -f "$PROXY_PIDFILE" ] && kill -0 "$(cat "$PROXY_PIDFILE")" 2>/dev/null; then
        if [ ! -f "$BORE_PIDFILE" ] || ! kill -0 "$(cat "$BORE_PIDFILE")" 2>/dev/null; then
            echo "[$(date -Iseconds)] BORE: dead, restarting" | tee -a "$LOG"
            start_bore
        fi
    fi

    sleep 30
done
