"""VoiceTools tests — no camera, no arm, no Gemma needed.

Patches the orchestrator's heavy entry points so the tool layer is
exercised in isolation. The same patterns Mattie + any other contributor
can use to add coverage for new skills.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from voice import tools as voice_tools


SAMPLE_SKILLS = {
    "pick_cutlery": {
        "description": "Pick up a piece of cutlery from the table",
        "preconditions": "Cutlery is on the table, gripper open",
        "postconditions": "Cutlery held, lifted clear",
        "episode_time_s": 15,
        "vla": {
            "backend": "smolvla",
            "uri": "Globalmysterysnailrevolution/xlerobot-pick-cutlery-smolvla",
        },
    },
    "pick_bowl": {
        "description": "Pick up the bowl from the table",
        "preconditions": "Bowl on table, gripper open",
        "postconditions": "Bowl held, lifted clear",
        "vla": {
            "backend": "smolvla",
            "uri": "Globalmysterysnailrevolution/xlerobot-pick-bowl-smolvla",
        },
    },
}


@pytest.fixture
def voice_tools_instance(monkeypatch: pytest.MonkeyPatch):
    """Construct VoiceTools with a stubbed skills catalog so unit tests
    don't need the real skills.yaml on disk."""
    with patch.object(voice_tools.orch, "load_skills", return_value=SAMPLE_SKILLS):
        t = voice_tools.VoiceTools()
    yield t


# ── list_skills ────────────────────────────────────────────────────────────


async def test_list_skills_returns_yaml_catalog(voice_tools_instance) -> None:
    out = await voice_tools_instance.list_skills()
    names = {s["name"] for s in out["skills"]}
    assert names == {"pick_cutlery", "pick_bowl"}
    cutlery = next(s for s in out["skills"] if s["name"] == "pick_cutlery")
    assert "Globalmysterysnailrevolution" in cutlery["vla_uri"]
    assert cutlery["vla_backend"] == "smolvla"


# ── look_at_scene ──────────────────────────────────────────────────────────


async def test_look_at_scene_calls_capture_and_describe(voice_tools_instance) -> None:
    fake_frame = b"\xff\xd8fakejpeg\xff\xd9"
    with patch.object(voice_tools.orch, "capture_frame", return_value=fake_frame) as cap, \
         patch.object(voice_tools.orch, "_gemma_call", return_value={"description": "A bowl sits next to a fork."}) as gem:
        out = await voice_tools_instance.look_at_scene(focus="the bowl")

    cap.assert_called_once_with(voice_tools.orch.CAMERA_INDEX)
    assert gem.call_count == 1
    args, _ = gem.call_args
    # _gemma_call(images, prompt, ...) — first arg is the image list
    assert args[0] == [fake_frame]
    assert "Focus on the bowl" in args[1]
    assert out["description"] == "A bowl sits next to a fork."


async def test_look_at_scene_handles_camera_failure(voice_tools_instance) -> None:
    def _boom(_idx: int) -> bytes:
        raise RuntimeError("camera busy")

    with patch.object(voice_tools.orch, "capture_frame", side_effect=_boom):
        out = await voice_tools_instance.look_at_scene()
    assert "error" in out
    assert "camera busy" in out["error"]
    assert out["where"] == "look_at_scene"


async def test_look_at_scene_handles_gemma_failure(voice_tools_instance) -> None:
    with patch.object(voice_tools.orch, "capture_frame", return_value=b"\xff\xd8"), \
         patch.object(voice_tools.orch, "_gemma_call", side_effect=RuntimeError("ollama down")):
        out = await voice_tools_instance.look_at_scene()
    assert "error" in out
    assert "ollama down" in out["error"]


# ── propose_skill ──────────────────────────────────────────────────────────


async def test_propose_skill_returns_planner_choice(voice_tools_instance) -> None:
    with patch.object(voice_tools.orch, "capture_frame", return_value=b"\xff\xd8"), \
         patch.object(
            voice_tools.orch,
            "select_next_skill",
            return_value={"done": False, "skill": "pick_cutlery", "reason": "Fork on table"},
         ):
        out = await voice_tools_instance.propose_skill(goal="set the table")
    assert out["done"] is False
    assert out["skill"] == "pick_cutlery"
    assert "Fork on table" in out["reason"]
    assert out["goal"] == "set the table"


# ── execute_skill ──────────────────────────────────────────────────────────


async def test_execute_skill_records_history(voice_tools_instance) -> None:
    with patch.object(
        voice_tools.orch,
        "execute_skill_with_verification",
        return_value=("completed", "Bowl is in the gripper"),
    ) as exe:
        out = await voice_tools_instance.execute_skill(skill="pick_bowl")

    exe.assert_called_once()
    name_arg, meta_arg, backend_arg = exe.call_args.args
    assert name_arg == "pick_bowl"
    assert meta_arg == SAMPLE_SKILLS["pick_bowl"]
    assert backend_arg == "smolvla"

    assert out["skill"] == "pick_bowl"
    assert out["verdict"] == "completed"
    assert out["reason"] == "Bowl is in the gripper"
    assert out["history_index"] == 0

    status = await voice_tools_instance.get_status()
    assert status["executions_total"] == 1
    assert status["last_execution"]["verdict"] == "completed"


async def test_execute_skill_unknown_returns_error(voice_tools_instance) -> None:
    out = await voice_tools_instance.execute_skill(skill="kiss_the_chef")
    assert "error" in out
    assert "unknown_skill" in out["error"]


async def test_execute_skill_handles_orchestrator_crash(voice_tools_instance) -> None:
    with patch.object(
        voice_tools.orch,
        "execute_skill_with_verification",
        side_effect=RuntimeError("lerobot-record exited 127"),
    ):
        out = await voice_tools_instance.execute_skill(skill="pick_bowl")
    assert "error" in out
    assert "lerobot-record" in out["error"]
    # Failed runs are still recorded so the history stays accurate.
    status = await voice_tools_instance.get_status()
    assert status["executions_total"] == 1
    assert status["last_execution"]["verdict"] == "error"


# ── queue_skills ───────────────────────────────────────────────────────────


async def test_queue_skills_runs_in_order(voice_tools_instance) -> None:
    with patch.object(
        voice_tools.orch,
        "execute_skill_with_verification",
        return_value=("completed", "ok"),
    ) as exe:
        out = await voice_tools_instance.queue_skills(
            steps=[{"skill": "pick_cutlery"}, {"skill": "pick_bowl"}]
        )
    assert out["completed"] == 2
    assert out["stopped_at"] is None
    assert exe.call_count == 2


async def test_queue_skills_bails_on_problem(voice_tools_instance) -> None:
    side_effects = [("completed", "ok"), ("problem", "dropped it")]
    with patch.object(
        voice_tools.orch,
        "execute_skill_with_verification",
        side_effect=side_effects,
    ):
        out = await voice_tools_instance.queue_skills(
            steps=[{"skill": "pick_cutlery"}, {"skill": "pick_bowl"}]
        )
    assert out["completed"] == 2  # both attempted
    assert out["stopped_at"] == 1
    assert "verdict=problem" in out["stopped_reason"]


async def test_queue_skills_cancelled_before_start(voice_tools_instance) -> None:
    await voice_tools_instance.cancel_current()
    with patch.object(
        voice_tools.orch, "execute_skill_with_verification"
    ) as exe:
        out = await voice_tools_instance.queue_skills(
            steps=[{"skill": "pick_bowl"}]
        )
    assert out["completed"] == 0
    assert out["stopped_reason"] == "cancelled"
    exe.assert_not_called()


# ── get_status ─────────────────────────────────────────────────────────────


async def test_get_status_empty(voice_tools_instance) -> None:
    out = await voice_tools_instance.get_status()
    assert out["executions_total"] == 0
    assert out["last_execution"] is None
    assert out["history"] == []
