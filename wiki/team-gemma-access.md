# Team access to the shared Gemma proxy

The pipeline runs Gemma-3 27B on Spark and exposes it through a bearer-token
proxy so any teammate can hit it without ssh access to Spark or sharing
credentials. Two access paths.

---

## Path A — On-LAN / Tailscale (recommended, lowest latency)

Latency: ~35 ms tablet → Spark. Works for anyone on the same Tailscale tailnet
or the same Wi-Fi as Spark.

**Setup once:**

1. Install Tailscale (https://tailscale.com/download)
2. Get an invite to the tailnet — ask the operator to send you a node-add link
3. Verify Spark is visible:
   ```bash
   tailscale status | grep spark
   # should show: 100.127.125.42  spark  ...
   ```

**Use:**

```bash
export OLLAMA_HOST="http://100.127.125.42:18080"
export GEMMA_PROXY_TOKEN="<paste from operator>"

# verify
curl -sH "Authorization: Bearer ${GEMMA_PROXY_TOKEN}" \
  "${OLLAMA_HOST}/api/tags" | jq .
```

---

## Path B — Public via bore tunnel (fallback when off-LAN)

Latency: 1-3 sec, sometimes flaky. Use only when you can't be on Tailscale.
URL changes on every bore restart (we keep `bore.pub:<port>` posted in the
team chat).

**Use:**

```bash
export OLLAMA_HOST="http://bore.pub:<current-port>"   # see team chat
export GEMMA_PROXY_TOKEN="<paste from operator>"

curl -sH "Authorization: Bearer ${GEMMA_PROXY_TOKEN}" \
  "${OLLAMA_HOST}/api/tags" | jq .
```

If the URL is stale or the request hangs, ping the operator — bore quick
tunnels die occasionally and the watchdog restarts them with a new port.

---

## What the proxy allows

The proxy (`scripts/spark/gemma_proxy.py`) restricts access tightly. Any
holder of the bearer token can:

- `POST /api/generate` with model `gemma3:27b` only
- `POST /api/chat` with model `gemma3:27b` only
- `GET /api/tags` (returns just `gemma3:27b`)

Any holder of the bearer token CANNOT:

- Pull/delete other models (`/api/pull`, `/api/delete` → 403)
- Use a different model (other names → 403)
- Access Spark over ssh (proxy lives in user-space, no root required)
- Touch any file outside `~/gemma-proxy/`

---

## Operator playbook

### Mint or rotate token

```bash
./scripts/deploy_gemma_proxy.sh                # generates fresh token
./scripts/rotate_gemma_token.sh                # convenience for rotating
```

The script prints the new token to stdout. Distribute via team chat (DM,
not channel — token gives Gemma access).

### Start the watchdog (auto-restart)

The watchdog keeps the proxy AND bore tunnel alive. Run on Spark:

```bash
ssh dgx-spark
nohup ~/gemma-proxy/proxy_watchdog.sh >/dev/null 2>&1 &
echo $! > ~/gemma-proxy/watchdog.pid
```

Logs to `~/gemma-proxy/watchdog.log`. Polls every 30 s. Restarts whatever
died with the original env (token preserved across restarts via
`~/gemma-proxy/proxy.env`).

### Check current bore URL

```bash
ssh dgx-spark "sed -E 's/\x1b\[[0-9;]*m//g' ~/gemma-proxy/bore.log | grep -oE 'bore\\.pub:[0-9]+' | tail -1"
```

Post the result in team chat whenever it changes (after watchdog restarts
bore — usually every few hours under load).

### Stop watchdog (and proxy + bore go quiet next 30s)

```bash
ssh dgx-spark 'kill $(cat ~/gemma-proxy/watchdog.pid 2>/dev/null) 2>/dev/null'
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 missing or invalid bearer token` | wrong token | check the value; ask operator to rotate if you suspect leak |
| `403 only model='gemma3:27b' is exposed` | sent a different model in the request body | set `model: "gemma3:27b"` in your request |
| `403 path /api/pull not allowed` | trying a blocked endpoint | only `/api/generate`, `/api/chat`, `/api/tags` exposed |
| Connection times out via bore | bore quick tunnel died | wait for watchdog to restart (≤30s), grab new URL from operator, OR switch to Tailscale |
| Connection times out via Tailscale | not on the tailnet, or Spark is offline | run `tailscale status`, ask operator to invite you, or fall back to bore |
