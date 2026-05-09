"""Bearer-authenticated reverse proxy for a single Ollama model.

Runs on Spark. Listens on $PROXY_PORT (default 18080), forwards a tightly
restricted subset of the Ollama API to localhost:11434:

  POST /api/generate   — must include "model": "<ALLOWED_MODEL>"
  POST /api/chat       — must include "model": "<ALLOWED_MODEL>"
  GET  /api/tags       — returns ONLY the allowed model entry

Every other path returns 403. There is no way through this proxy to:
  - pull or download new models   (POST /api/pull blocked)
  - delete a model                (DELETE /api/delete blocked)
  - push a model off Spark        (POST /api/push blocked)
  - access the host filesystem    (no static file routes)

Auth: Bearer token in `Authorization: Bearer <token>` header, compared against
$PROXY_TOKEN. Constant-time comparison via hmac.compare_digest. No fallback,
no anonymous mode.

Env:
  PROXY_TOKEN    REQUIRED. Bearer token. ~32 url-safe bytes recommended.
  PROXY_PORT     Default 18080.
  PROXY_BIND     Default 0.0.0.0 (LAN-reachable). Use 127.0.0.1 to disable.
  ALLOWED_MODEL  Default gemma3:27b.
  OLLAMA_URL     Default http://127.0.0.1:11434.

Usage on Spark:
  PROXY_TOKEN=$(openssl rand -hex 32) nohup python3 gemma_proxy.py \\
    > ~/gemma-proxy/proxy.log 2>&1 &

Stdlib-only — no pip install required on Spark.
"""
from __future__ import annotations

import hmac
import http.server
import json
import os
import socketserver
import sys
import urllib.error
import urllib.request

PROXY_TOKEN = os.environ.get("PROXY_TOKEN")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "18080"))
PROXY_BIND = os.environ.get("PROXY_BIND", "0.0.0.0")
ALLOWED_MODEL = os.environ.get("ALLOWED_MODEL", "gemma3:27b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

if not PROXY_TOKEN:
    print("FATAL: PROXY_TOKEN env var not set", file=sys.stderr)
    sys.exit(1)
if len(PROXY_TOKEN) < 16:
    print("FATAL: PROXY_TOKEN too short (min 16 chars)", file=sys.stderr)
    sys.exit(1)

ALLOWED_PATHS_POST = {"/api/generate", "/api/chat"}
ALLOWED_PATHS_GET = {"/api/tags"}


def _check_auth(handler: http.server.BaseHTTPRequestHandler) -> bool:
    auth = handler.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth[7:].strip(), PROXY_TOKEN)


def _send_json(handler, status: int, body: dict) -> None:
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _forward(handler, method: str, body: bytes | None) -> None:
    """Forward the (already authenticated, already validated) request to Ollama."""
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(
        f"{OLLAMA_URL}{handler.path}",
        data=body,
        method=method,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
            handler.send_response(resp.status)
            handler.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        handler.send_response(e.code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except Exception as e:
        _send_json(handler, 502, {"error": "upstream", "detail": str(e)})


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "GemmaProxy/1.0"

    def log_message(self, fmt, *args):
        # Redact Authorization from logs; keep method+path+status.
        sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

    def do_POST(self):
        if not _check_auth(self):
            _send_json(self, 401, {"error": "missing or invalid bearer token"})
            return
        if self.path not in ALLOWED_PATHS_POST:
            _send_json(self, 403, {"error": f"path {self.path} not allowed"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 50 * 1024 * 1024:
            _send_json(self, 413, {"error": "body missing or too large (>50MB)"})
            return
        body = self.rfile.read(length)

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            _send_json(self, 400, {"error": "body must be JSON"})
            return

        model = payload.get("model")
        if model != ALLOWED_MODEL:
            _send_json(self, 403, {"error": f"only model={ALLOWED_MODEL!r} is exposed via this proxy"})
            return

        _forward(self, "POST", body)

    def do_GET(self):
        if not _check_auth(self):
            _send_json(self, 401, {"error": "missing or invalid bearer token"})
            return
        if self.path not in ALLOWED_PATHS_GET:
            _send_json(self, 403, {"error": f"path {self.path} not allowed"})
            return

        # Forward, then filter the response so only ALLOWED_MODEL is visible.
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}{self.path}", timeout=10) as resp:
                upstream = json.loads(resp.read())
        except Exception as e:
            _send_json(self, 502, {"error": "upstream", "detail": str(e)})
            return

        if isinstance(upstream, dict) and "models" in upstream:
            upstream["models"] = [m for m in upstream["models"] if m.get("name") == ALLOWED_MODEL]
        _send_json(self, 200, upstream)

    def do_DELETE(self):  # noqa: N802 — http.server convention
        _send_json(self, 403, {"error": "DELETE not exposed"})

    def do_PUT(self):  # noqa: N802
        _send_json(self, 403, {"error": "PUT not exposed"})


class ThreadingHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    print(f"gemma-proxy: bind={PROXY_BIND}:{PROXY_PORT}  upstream={OLLAMA_URL}  model={ALLOWED_MODEL}")
    print(f"  paths: POST {sorted(ALLOWED_PATHS_POST)}  GET {sorted(ALLOWED_PATHS_GET)}")
    print(f"  token: <set, {len(PROXY_TOKEN)} chars>")
    server = ThreadingHTTP((PROXY_BIND, PROXY_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
