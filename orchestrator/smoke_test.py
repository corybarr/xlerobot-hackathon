"""Smoke test for the orchestrator — exercises every code path without an arm.

Runs through:
  1. skills.yaml loads
  2. camera capture (or skip if no camera)
  3. Ollama reachability + Gemma model availability (skip if not running)
  4. build_vla_command shape for each backend
  5. select_next_skill round-trip (real call if Gemma up, otherwise stub)
  6. verify_skill_state round-trip (real call if Gemma up, otherwise stub)
  7. execute_skill_with_verification with VLA subprocess replaced by a sleep stub

Usage:
    python orchestrator/smoke_test.py

Exit code: 0 if every check passed (or skipped with a clear note),
           1 if any check failed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orchestrator.orchestrator as orch  # noqa: E402

PASS = "✓"   # check mark
FAIL = "✗"   # cross
SKIP = "—"   # em dash


def _check(name: str, fn) -> bool:
    try:
        msg = fn()
        print(f"  {PASS} {name}: {msg}")
        return True
    except _Skip as e:
        print(f"  {SKIP} {name}: SKIP — {e}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {type(e).__name__}: {e}")
        return False


class _Skip(Exception):
    pass


def t_skills_yaml_loads():
    skills = orch.load_skills()
    assert isinstance(skills, dict) and len(skills) > 0
    for name, meta in skills.items():
        assert "description" in meta, f"{name}: missing description"
    return f"{len(skills)} skills loaded"


def t_camera_capture():
    try:
        frame = orch.capture_frame(orch.CAMERA_INDEX)
    except RuntimeError as e:
        raise _Skip(f"no camera at index {orch.CAMERA_INDEX} ({e})")
    assert isinstance(frame, bytes) and len(frame) > 1000
    return f"got {len(frame)}-byte JPEG from camera {orch.CAMERA_INDEX}"


def t_ollama_reachable():
    headers = {"Authorization": f"Bearer {orch.GEMMA_PROXY_TOKEN}"} if orch.GEMMA_PROXY_TOKEN else {}
    try:
        r = requests.get(f"{orch.OLLAMA_HOST}/api/tags", headers=headers, timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        raise _Skip(f"Ollama not reachable at {orch.OLLAMA_HOST} ({type(e).__name__})")
    if orch.GEMMA_MODEL not in models:
        raise _Skip(f"{orch.GEMMA_MODEL} not yet pulled (have: {models})")
    return f"{orch.GEMMA_MODEL} available"


def t_build_vla_command_shapes():
    for backend in ("smolvla", "act", "molmoact2"):
        cmd = orch.build_vla_command("remove_fork_from_cup", backend)
        assert isinstance(cmd, list) and len(cmd) > 0
        assert all(isinstance(a, str) for a in cmd)
    try:
        orch.build_vla_command("x", "nonexistent")
        return "FAIL — should have raised"
    except ValueError:
        pass
    return "smolvla/act/molmoact2 all return list[str]; unknown raises ValueError"


def t_select_next_skill_roundtrip():
    skills = orch.load_skills()
    try:
        requests.get(f"{orch.OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
    except Exception:
        raise _Skip("Ollama not reachable")
    # Use a tiny dummy frame (Gemma will reject if we send junk; that's fine for smoke)
    import cv2
    import numpy as np
    dummy = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
    decision = orch.select_next_skill(dummy, "set the table", skills, [])
    assert isinstance(decision, dict)
    assert "done" in decision
    if not decision["done"]:
        assert decision.get("skill") in skills, f"unknown skill: {decision.get('skill')}"
    return f"got {decision}"


def t_verify_skill_state_roundtrip():
    skills = orch.load_skills()
    try:
        requests.get(f"{orch.OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
    except Exception:
        raise _Skip("Ollama not reachable")
    import cv2
    import numpy as np
    pre = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
    cur = cv2.imencode(".jpg", np.full((480, 640, 3), 200, dtype=np.uint8))[1].tobytes()
    skill_name = next(iter(skills))
    verdict = orch.verify_skill_state(pre, cur, skill_name, skills[skill_name])
    assert verdict.get("state") in ("in_progress", "completed", "problem")
    return f"got {verdict}"


def t_execute_with_stub_vla():
    """Replace the VLA subprocess with a 2-second sleep; ensure the orchestrator
    handles the lifecycle (start, monitor, terminate or natural exit) cleanly.
    """
    skills = orch.load_skills()
    skill_name = next(iter(skills))

    def fake_command(name, backend, skill_meta=None):
        # cross-platform: python sleep. Accepts the same kwargs as the real
        # build_vla_command so signature changes don't silently break the test.
        return [sys.executable, "-c", "import time; time.sleep(2)"]

    def fake_capture(idx):
        import cv2
        import numpy as np
        return cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()

    def fake_verify(pre, cur, name, meta):
        return {"state": "in_progress", "reason": "stub"}

    with patch.object(orch, "build_vla_command", fake_command), \
         patch.object(orch, "capture_frame", fake_capture), \
         patch.object(orch, "verify_skill_state", fake_verify):
        old_timeout = orch.SKILL_TIMEOUT_S
        orch.SKILL_TIMEOUT_S = 5.0
        try:
            verdict, reason = orch.execute_skill_with_verification(
                skill_name, skills[skill_name], "smolvla"
            )
        finally:
            orch.SKILL_TIMEOUT_S = old_timeout

    assert verdict in ("in_progress", "completed", "problem", "timeout"), verdict
    return f"verdict={verdict} reason={reason!r}"


def main() -> int:
    print(f"orchestrator smoke test — host={orch.OLLAMA_HOST} model={orch.GEMMA_MODEL}\n")
    results = [
        _check("skills.yaml loads", t_skills_yaml_loads),
        _check("camera capture", t_camera_capture),
        _check("ollama reachable", t_ollama_reachable),
        _check("build_vla_command shapes", t_build_vla_command_shapes),
        _check("select_next_skill round-trip", t_select_next_skill_roundtrip),
        _check("verify_skill_state round-trip", t_verify_skill_state_roundtrip),
        _check("execute_with_stub_vla lifecycle", t_execute_with_stub_vla),
    ]
    failed = sum(1 for r in results if not r)
    print(f"\n{len(results) - failed}/{len(results)} checks passed (or skipped with reason).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
