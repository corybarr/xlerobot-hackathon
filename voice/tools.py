"""Voice-callable skills — one tool per trained VLA.

Architecture:

    Phone -> Vapi -> Gemma 4 (planner, on Spark) -> tool call to /api/tools/<skill>
                                                            |
                                              voice/server.py on tablet
                                                            |
                                                spawns lerobot-record with
                                                --policy.path=<HF repo>
                                                            |
                                                  Trained VLA drives the arm

Each ``pick_*`` tool corresponds to one entry in ``skills/skills.yaml``.
Gemma picks which to call; this module just runs the VLA. No verifier
loop, no orchestrator-style SELECT/INVOKE/VERIFY — Gemma's voice
conversation IS the loop.

``look_at_scene`` is the only multimodal tool: it captures a frame from
a separate "vision camera" (different index from the arm-recording
cameras) and asks Gemma what it sees. Vapi reads the description aloud.

The VLA invocation
------------------

We use ``lerobot-record`` from the lerobot conda env. In current lerobot
that's the de-facto inference path on real hardware: ``num_episodes=1``
+ ``push_to_hub=false`` runs the policy for one episode and discards the
local dataset. There's no separate ``lerobot-infer`` in the released
lerobot today; ``lerobot-eval`` is sim-only.

Live narration over the phone
-----------------------------

Each VLA run takes 15-30 s. The async pattern (``pick_*_async`` returns
a job_id; ``get_status(job_id)`` polls) keeps the caller from sitting
in silence. The system prompt teaches Gemma to call ``get_status`` every
few seconds and read the latest update aloud.
"""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import requests
import yaml
from loguru import logger


# ── Settings from env (no orchestrator dependency) ────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_YAML = REPO_ROOT / "skills" / "skills.yaml"

# Camera dedicated to Gemma's vision (separate from arm cameras).
# The arm-recording cameras are configured in lerobot-record's robot config;
# the *vision* camera here is just for look_at_scene.
VISION_CAMERA_INDEX = int(os.environ.get("VISION_CAMERA_INDEX", os.environ.get("CAMERA_INDEX", "0")))

# Arm ports — passed through to lerobot-record.
FOLLOWER_PORT = os.environ.get("FOLLOWER_PORT", "COM10")
LEADER_PORT = os.environ.get("LEADER_PORT", "COM7")

# How to reach Gemma (for look_at_scene's multimodal describe).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma3:27b")
GEMMA_PROXY_TOKEN = os.environ.get("GEMMA_PROXY_TOKEN", "")

# Where to find the lerobot CLI. Default to the conda env path the team
# uses; override LEROBOT_RECORD_BIN to point elsewhere.
LEROBOT_RECORD_BIN = os.environ.get(
    "LEROBOT_RECORD_BIN",
    str(Path.home() / "miniconda3" / "envs" / "lerobot" / "Scripts" / "lerobot-record.exe"),
)

SKILL_TIMEOUT_S = float(os.environ.get("SKILL_TIMEOUT_S", "45.0"))


def _load_skills() -> dict[str, dict[str, Any]]:
    """Read skills.yaml. The voice service must restart to pick up edits."""
    if not SKILLS_YAML.exists():
        logger.warning("skills.yaml not found at {} — VoiceTools will be empty.", SKILLS_YAML)
        return {}
    with open(SKILLS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Job machinery ─────────────────────────────────────────────────────────


@dataclass
class JobState:
    job_id: str
    skill: str
    started_at: float
    state: str = "starting"   # starting | running | completed | error | timeout | cancelled
    message: str = ""
    finished_at: Optional[float] = None
    proc_pid: Optional[int] = None
    log_tail: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "skill": self.skill,
            "state": self.state,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 2),
            "log_tail": self.log_tail[-5:],
        }


# ── VoiceTools ────────────────────────────────────────────────────────────


class VoiceTools:
    """Skill-shaped tools Gemma calls via Vapi.

    The instance is built once at server startup. One arm, one camera —
    we serialize concurrent skill executions through ``_lock``.
    """

    def __init__(self) -> None:
        self._skills: dict[str, dict[str, Any]] = _load_skills()
        self._jobs: dict[str, JobState] = {}
        self._current_proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._cancel_flag = False

    # ── look_at_scene ────────────────────────────────────────────────

    async def look_at_scene(self, focus: str = "") -> dict[str, Any]:
        """Capture the vision camera and have Gemma describe what's there.

        Uses a different camera index from the arm-recording cameras —
        Gemma sees the table from its own viewpoint. Set ``VISION_CAMERA_INDEX``.
        """
        loop = asyncio.get_running_loop()
        try:
            jpeg = await loop.run_in_executor(None, _grab_frame, VISION_CAMERA_INDEX)
        except Exception as exc:
            logger.warning("camera grab failed: {}", exc)
            return {"error": f"camera_failed: {exc!s}", "where": "look_at_scene"}

        focus_clause = f" Focus on {focus}." if focus else ""
        prompt = (
            "Describe what you see on the table in one or two short, "
            "conversational sentences. Name objects you actually see "
            f"(cup, bowl, cutlery, etc.). Under 35 words.{focus_clause}"
        )

        try:
            description = await loop.run_in_executor(None, _gemma_describe, jpeg, prompt)
        except Exception as exc:
            logger.warning("gemma describe failed: {}", exc)
            return {"error": f"gemma_failed: {exc!s}", "where": "look_at_scene"}

        return {"description": description, "captured_at": time.time()}

    # ── list_skills ──────────────────────────────────────────────────

    async def list_skills(self) -> dict[str, Any]:
        """All skills from skills.yaml in a stable format."""
        items = []
        for name, meta in self._skills.items():
            items.append({
                "name": name,
                "description": meta.get("description", ""),
                "preconditions": meta.get("preconditions", ""),
                "postconditions": meta.get("postconditions", ""),
                "vla_uri": (meta.get("vla") or {}).get("uri"),
                "episode_time_s": meta.get("episode_time_s"),
            })
        return {"skills": items, "count": len(items)}

    # ── per-skill convenience tools (one-call dispatch) ──────────────

    async def pick_cup(self) -> dict[str, Any]:
        """Run the trained cup-pick VLA on the arm."""
        return await self._run_skill("pick_cup")

    async def pick_bowl(self) -> dict[str, Any]:
        """Run the trained bowl-pick VLA on the arm."""
        return await self._run_skill("pick_bowl")

    async def pick_cutlery(self) -> dict[str, Any]:
        """Run the trained cutlery-pick VLA on the arm."""
        return await self._run_skill("pick_cutlery")

    async def pick_anything(self) -> dict[str, Any]:
        """Generalist fallback — Mattie's 3-cam fine-tune."""
        return await self._run_skill("pick_anything")

    async def run_skill(self, skill: str) -> dict[str, Any]:
        """Run ANY skill by name. Use this instead of `pick_*` if Gemma
        prefers a single tool with an argument."""
        return await self._run_skill(skill)

    # ── get_status / cancel ──────────────────────────────────────────

    async def get_status(self, job_id: str = "") -> dict[str, Any]:
        """Without ``job_id``: a snapshot of the most-recent / active job.
        With ``job_id``: that specific job's state."""
        if job_id:
            job = self._jobs.get(job_id)
            if job is None:
                return {"error": f"unknown_job: {job_id!r}", "where": "get_status"}
            return job.snapshot()

        active = [j for j in self._jobs.values() if j.finished_at is None]
        if active:
            return active[0].snapshot()
        if self._jobs:
            most_recent = max(self._jobs.values(), key=lambda j: j.started_at)
            return most_recent.snapshot()
        return {"state": "idle", "message": "no jobs yet"}

    async def cancel_current(self) -> dict[str, Any]:
        """Best-effort kill any running VLA subprocess."""
        self._cancel_flag = True
        proc = self._current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return {"ok": True, "killed_pid": proc.pid}
            except Exception as exc:
                return {"error": f"kill_failed: {exc!s}", "where": "cancel_current"}
        return {"ok": True, "killed_pid": None, "note": "nothing was running"}

    # ── internal ─────────────────────────────────────────────────────

    async def _run_skill(self, skill: str) -> dict[str, Any]:
        """Spawn lerobot-record for one skill, return a job_id immediately.

        The agent polls ``get_status(job_id)`` to read live updates as
        the VLA runs. Skill execution is single-threaded (one arm, one
        camera) — concurrent calls return an error.
        """
        meta = self._skills.get(skill)
        if not meta:
            return {
                "error": f"unknown_skill: {skill!r}. Known: {sorted(self._skills)}",
                "where": "_run_skill",
            }
        vla = meta.get("vla") or {}
        uri = vla.get("uri")
        if not uri:
            return {"error": f"no_vla_uri_for: {skill!r}", "where": "_run_skill"}

        with self._lock:
            if self._current_proc is not None and self._current_proc.poll() is None:
                return {
                    "error": "another_skill_running",
                    "where": "_run_skill",
                    "hint": "call cancel_current first, or wait for the running job to finish",
                }
            self._cancel_flag = False
            job = JobState(
                job_id=uuid.uuid4().hex[:8],
                skill=skill,
                started_at=time.time(),
            )
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._exec_thread,
            args=(job, uri, int(meta.get("episode_time_s") or 15)),
            name=f"vla-{job.job_id}",
            daemon=True,
        )
        thread.start()

        return {
            "job_id": job.job_id,
            "skill": skill,
            "vla_uri": uri,
            "started_at": job.started_at,
            "hint": "Call get_status(job_id=...) every ~4 seconds to narrate progress.",
        }

    def _exec_thread(self, job: JobState, vla_uri: str, episode_time_s: int) -> None:
        """Run lerobot-record in a background thread and stream logs into the job."""
        cmd = [
            LEROBOT_RECORD_BIN,
            "--robot.type=so101_follower",
            f"--robot.port={FOLLOWER_PORT}",
            "--robot.id=arm_a_follower",
            "--teleop.type=so101_leader",
            f"--teleop.port={LEADER_PORT}",
            "--teleop.id=arm_a_leader",
            f"--policy.path={vla_uri}",
            "--dataset.repo_id=local-eval",
            "--dataset.num_episodes=1",
            f"--dataset.episode_time_s={episode_time_s}",
            "--dataset.push_to_hub=false",
        ]
        job.state = "running"
        job.message = f"launching {Path(vla_uri).name}"
        job.log_tail.append(job.message)
        logger.info("[{}] $ {}", job.job_id, " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            job.state = "error"
            job.message = (
                f"lerobot-record not found at {LEROBOT_RECORD_BIN}. "
                "Set LEROBOT_RECORD_BIN env var."
            )
            job.log_tail.append(job.message)
            job.finished_at = time.time()
            return
        except Exception as exc:
            job.state = "error"
            job.message = f"spawn_failed: {exc!s}"
            job.log_tail.append(job.message)
            job.finished_at = time.time()
            return

        with self._lock:
            self._current_proc = proc
            job.proc_pid = proc.pid

        deadline = time.time() + SKILL_TIMEOUT_S
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    job.log_tail.append(line)
                    # keep the message field current with whatever's recent
                    job.message = line[-180:]
                if self._cancel_flag:
                    proc.terminate()
                    job.state = "cancelled"
                    job.message = "user cancelled"
                    break
                if time.time() > deadline:
                    proc.terminate()
                    job.state = "timeout"
                    job.message = f"exceeded {SKILL_TIMEOUT_S}s"
                    break
            proc.wait(timeout=10)
        except Exception as exc:
            logger.warning("[{}] stream-read failed: {}", job.job_id, exc)
            job.state = "error"
            job.message = f"stream_failed: {exc!s}"
        finally:
            if job.state == "running":
                job.state = "completed" if proc.returncode == 0 else "error"
                job.message = (
                    "VLA finished cleanly"
                    if proc.returncode == 0
                    else f"lerobot-record exited {proc.returncode}"
                )
            job.finished_at = time.time()
            with self._lock:
                if self._current_proc is proc:
                    self._current_proc = None
                self._cancel_flag = False


# ── Helpers (no orchestrator dependency) ──────────────────────────────────


def _grab_frame(camera_index: int) -> bytes:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"vision camera {camera_index} returned no frame")
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed on captured frame")
    return jpeg.tobytes()


def _gemma_describe(jpeg: bytes, prompt: str) -> str:
    """POST one image + prompt to the Ollama / gemma-proxy and return text."""
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(jpeg).decode("ascii")],
        "stream": False,
    }
    headers = (
        {"Authorization": f"Bearer {GEMMA_PROXY_TOKEN}"}
        if GEMMA_PROXY_TOKEN
        else {}
    )
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate", json=payload, headers=headers, timeout=30
    )
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


__all__ = ["VoiceTools", "JobState"]
