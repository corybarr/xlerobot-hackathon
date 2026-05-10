"""Voice-callable tool layer. Every tool wraps an orchestrator function.

The voice agent's tool surface is intentionally tiny — the orchestrator
already does the SELECT / INVOKE / VERIFY work, this module just exposes
it as async tools for Pipecat to dispatch.

Tools:

* ``look_at_scene(focus?)``  — capture a frame + ask Gemma for a description
* ``list_skills()``           — return skills.yaml as a structured catalog
* ``propose_skill(goal?)``    — orchestrator.select_next_skill (Gemma picks)
* ``execute_skill(name)``     — orchestrator.execute_skill_with_verification
                                 (runs lerobot-record subprocess + Gemma verifier)
* ``queue_skills(steps)``     — execute a list of skills in order
* ``get_status()``            — last action outcome + history snapshot
* ``cancel_current()``        — terminate any running VLA subprocess

Every method returns a JSON-serialisable dict. Errors never raise — they
come back as ``{"error": str, "where": str}`` so the voice loop keeps
running and the human gets to hear what went wrong.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

# Reuse the orchestrator's plumbing wholesale.
from orchestrator import orchestrator as orch


@dataclass
class _ExecRecord:
    when: float
    skill: str
    verdict: str
    reason: str
    duration_s: float


class VoiceTools:
    """Async wrappers around orchestrator functions.

    One instance per server process — the orchestrator's subprocess + camera
    state aren't safe for concurrent skill execution. The Pipecat pipeline
    serializes through ``_executor_lock``.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Any] = orch.load_skills()
        self._history: list[_ExecRecord] = []
        self._current_proc: Optional[Any] = None
        self._executor_lock = asyncio.Lock()
        self._cancel_requested = False
        # Cache the last frame so look_at_scene can return promptly even if
        # the camera is briefly busy. Refreshed every time we actually look.
        self._last_frame_b64: Optional[str] = None
        self._last_frame_ts: float = 0.0

    # ------------------------------------------------------------------
    # look_at_scene  — Gemma multimodal "what do you see"
    # ------------------------------------------------------------------

    async def look_at_scene(self, focus: str = "") -> dict[str, Any]:
        """Capture the current camera frame and have Gemma describe it.

        Reuses :func:`orchestrator.orchestrator.capture_frame` for the
        camera grab and :func:`_gemma_call` for the multimodal call — same
        proxy, same auth, same model as the orchestrator. The voice loop
        gets the description as plain text it can speak.
        """
        loop = asyncio.get_running_loop()
        try:
            frame_bytes: bytes = await loop.run_in_executor(
                None, orch.capture_frame, orch.CAMERA_INDEX
            )
        except Exception as exc:
            logger.warning("capture_frame failed: {}", exc)
            return {"error": f"camera_failed: {exc!s}", "where": "look_at_scene"}

        self._last_frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
        self._last_frame_ts = time.time()

        focus_clause = f" Focus on {focus}." if focus else ""
        prompt = (
            "Describe what you see on the table in one or two short, "
            "conversational sentences. Name the objects you actually see "
            "(cup, bowl, cutlery, etc.). Under 35 words." + focus_clause
        )

        def _ask() -> dict[str, Any]:
            try:
                resp = orch._gemma_call([frame_bytes], prompt, timeout=30)  # noqa: SLF001
                return resp if isinstance(resp, dict) else {"raw": str(resp)}
            except Exception as exc:
                return {"error": f"gemma_describe_failed: {exc!s}"}

        result = await loop.run_in_executor(None, _ask)

        # orchestrator._gemma_call returns parsed JSON because we ask the
        # model for "format": "json". For a free-text describe, we override
        # to plain text — but the existing helper wraps it. Either way,
        # surface whatever we get.
        description = ""
        if isinstance(result, dict):
            if "error" in result:
                return {"error": result["error"], "where": "look_at_scene"}
            description = (
                result.get("description")
                or result.get("text")
                or result.get("raw")
                or str(result)
            )
        return {
            "description": str(description).strip(),
            "captured_at": self._last_frame_ts,
        }

    # ------------------------------------------------------------------
    # list_skills  — feed Gemma the catalog from skills.yaml
    # ------------------------------------------------------------------

    async def list_skills(self) -> dict[str, Any]:
        """Return every skill from ``skills/skills.yaml`` plus its VLA target.

        Stable for the call; the LLM caches this. Disabled skills (commented
        out in YAML) don't appear because ``yaml.safe_load`` doesn't see them.
        """
        items = []
        for name, meta in self._skills.items():
            items.append({
                "name": name,
                "description": meta.get("description", ""),
                "preconditions": meta.get("preconditions", ""),
                "postconditions": meta.get("postconditions", ""),
                "vla_uri": (meta.get("vla") or {}).get("uri"),
                "vla_backend": (meta.get("vla") or {}).get("backend", "smolvla"),
                "episode_time_s": meta.get("episode_time_s"),
            })
        return {"skills": items, "count": len(items)}

    # ------------------------------------------------------------------
    # propose_skill  — orchestrator.select_next_skill (Gemma picks)
    # ------------------------------------------------------------------

    async def propose_skill(self, goal: str = "") -> dict[str, Any]:
        """Have Gemma pick the next skill based on the current scene + history.

        Delegates to :func:`orchestrator.orchestrator.select_next_skill`.
        Returns ``{"done": bool, "skill": str?, "reason": str}`` exactly as
        the orchestrator format. The voice agent reads the reason aloud.
        """
        loop = asyncio.get_running_loop()
        try:
            frame_bytes: bytes = await loop.run_in_executor(
                None, orch.capture_frame, orch.CAMERA_INDEX
            )
        except Exception as exc:
            return {"error": f"camera_failed: {exc!s}", "where": "propose_skill"}

        # Translate our history records into the orchestrator's StepRecord
        # shape so select_next_skill can render the "history (most recent
        # last)" block exactly the way it expects.
        history = [
            orch.StepRecord(
                skill=h.skill,
                select_reason="",  # unknown after-the-fact
                verdict=h.verdict,
                verify_reason=h.reason,
            )
            for h in self._history
        ]
        actual_goal = goal or orch.GOAL

        try:
            choice = await loop.run_in_executor(
                None,
                orch.select_next_skill,
                frame_bytes,
                actual_goal,
                self._skills,
                history,
            )
        except Exception as exc:
            return {"error": f"select_failed: {exc!s}", "where": "propose_skill"}

        return {
            "done": bool(choice.get("done")),
            "skill": choice.get("skill"),
            "reason": choice.get("reason", ""),
            "goal": actual_goal,
        }

    # ------------------------------------------------------------------
    # execute_skill  — orchestrator.execute_skill_with_verification
    # ------------------------------------------------------------------

    async def execute_skill(self, *, skill: str) -> dict[str, Any]:
        """Run one named skill end-to-end via lerobot-record.

        Calls :func:`orchestrator.orchestrator.execute_skill_with_verification`
        which:

          1. captures a pre-frame
          2. spawns ``lerobot-record --policy.path=<HF repo>``
          3. periodically asks Gemma to compare pre vs current and classify
             in_progress / completed / problem
          4. terminates the VLA on completed / problem / timeout

        Returns ``{"skill": str, "verdict": str, "reason": str, "took_seconds":
        float, "history_index": int}``.
        """
        if skill not in self._skills:
            return {
                "error": f"unknown_skill: {skill!r}. Known: {sorted(self._skills)}",
                "where": "execute_skill",
            }

        async with self._executor_lock:
            if self._cancel_requested:
                self._cancel_requested = False
                return {"error": "cancelled_before_start", "where": "execute_skill"}

            meta = self._skills[skill]
            backend = (meta.get("vla") or {}).get("backend", "smolvla")
            start = time.time()
            loop = asyncio.get_running_loop()
            try:
                verdict_tuple = await loop.run_in_executor(
                    None,
                    orch.execute_skill_with_verification,
                    skill,
                    meta,
                    backend,
                )
            except Exception as exc:
                logger.warning("execute_skill_with_verification crashed: {}", exc)
                record = _ExecRecord(
                    when=time.time(),
                    skill=skill,
                    verdict="error",
                    reason=str(exc),
                    duration_s=time.time() - start,
                )
                self._history.append(record)
                return {
                    "error": f"execute_failed: {exc!s}",
                    "where": "execute_skill",
                    "skill": skill,
                }

            verdict, reason = verdict_tuple
            duration = time.time() - start
            record = _ExecRecord(
                when=time.time(),
                skill=skill,
                verdict=verdict,
                reason=reason,
                duration_s=duration,
            )
            self._history.append(record)

            return {
                "skill": skill,
                "verdict": verdict,
                "reason": reason,
                "took_seconds": round(duration, 2),
                "history_index": len(self._history) - 1,
                "vla_uri": (meta.get("vla") or {}).get("uri"),
                "backend": backend,
            }

    # ------------------------------------------------------------------
    # queue_skills  — execute a list in order, bail on first failure
    # ------------------------------------------------------------------

    async def queue_skills(self, *, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Run a list of skills sequentially.

        Each step is an ``execute_skill`` kwargs dict, e.g.
        ``[{"skill": "pick_cutlery"}, {"skill": "pick_bowl"}]``. Bails on the
        first non-``completed`` verdict.
        """
        results: list[dict[str, Any]] = []
        for idx, step in enumerate(steps):
            if self._cancel_requested:
                self._cancel_requested = False
                return {
                    "completed": idx,
                    "results": results,
                    "stopped_at": idx,
                    "stopped_reason": "cancelled",
                }
            out = await self.execute_skill(**step)
            results.append(out)
            verdict = out.get("verdict")
            if out.get("error") or verdict not in ("completed", None):
                return {
                    "completed": idx + 1,
                    "results": results,
                    "stopped_at": idx,
                    "stopped_reason": out.get("error") or f"verdict={verdict}",
                }
        return {
            "completed": len(results),
            "results": results,
            "stopped_at": None,
            "stopped_reason": None,
        }

    # ------------------------------------------------------------------
    # get_status / cancel
    # ------------------------------------------------------------------

    async def get_status(self) -> dict[str, Any]:
        """Snapshot of the action history + last frame age. No side effects."""
        last = self._history[-1] if self._history else None
        return {
            "executions_total": len(self._history),
            "last_execution": (
                {
                    "when": last.when,
                    "skill": last.skill,
                    "verdict": last.verdict,
                    "reason": last.reason,
                    "duration_s": round(last.duration_s, 2),
                }
                if last is not None
                else None
            ),
            "last_frame_age_s": (
                round(time.time() - self._last_frame_ts, 1) if self._last_frame_ts else None
            ),
            "history": [
                {"skill": h.skill, "verdict": h.verdict, "reason": h.reason}
                for h in self._history[-5:]
            ],
        }

    async def cancel_current(self) -> dict[str, Any]:
        """Best-effort interrupt for the next queued step.

        A skill already in flight runs to completion / problem / timeout —
        the orchestrator's own SKILL_TIMEOUT_S is the upper bound. For
        emergency stops on the arm side, the user can power-cycle COM10.
        """
        self._cancel_requested = True
        return {"ok": True, "cancel_armed": True}


__all__ = ["VoiceTools"]
