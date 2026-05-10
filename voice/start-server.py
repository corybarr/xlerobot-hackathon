"""Spawn voice/server.py in a NEW console window (Windows).

uvicorn exits silently when stdout is a redirected pipe (we hit this
twice during development). Allocating a real console via
``CREATE_NEW_CONSOLE`` sidesteps the issue. Double-click ``run-server.cmd``
to use the .cmd entry-point, or run::

    python start-server.py

The new console window stays open even after this launcher exits.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
VENV_PY = HERE / ".venv" / "Scripts" / "python.exe"

if not VENV_PY.is_file():
    print(f"ERROR: venv python not found at {VENV_PY}", file=sys.stderr)
    print("Run:  python -m venv .venv && .venv/Scripts/python -m pip install -e .[dev]", file=sys.stderr)
    sys.exit(1)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

CREATE_NEW_CONSOLE = 0x00000010  # only on Windows

print(f"==> launching {VENV_PY} -u -m voice.server (new console)")
proc = subprocess.Popen(
    [str(VENV_PY), "-u", "-m", "voice.server"],
    cwd=str(HERE),
    env=env,
    creationflags=CREATE_NEW_CONSOLE if os.name == "nt" else 0,
)
print(f"    pid: {proc.pid}")
print("    server logs land in the new console window — close it to stop.")
print("    smoke:  curl http://localhost:8765/health")
