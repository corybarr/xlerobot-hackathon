"""Inference API endpoints."""

import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.models.inference import (
    InferenceRequest,
    InferenceResponse,
    MolmoLoadRequest,
    MolmoStartRequest,
)
from backend.models.system import ProcessStatus
from backend.services.config_manager import ConfigManager
from backend.services.port_lock_manager import PortInUseError, port_lock_manager
from backend.services.process_manager import process_manager

from backend.inference import (
    DEFAULT_MOLMO_MODEL_ID,
    InferenceLoop,
    get_inference_loop,
    get_molmo_engine,
    set_inference_loop,
    start_background_load,
)

router = APIRouter()
config_manager = ConfigManager()
logger = logging.getLogger(__name__)


def _molmo_not_loaded() -> None:
    raise HTTPException(status_code=400, detail={"error": "model not loaded"})


def build_inference_command(config, request: InferenceRequest) -> list[str]:
    """Build inference command from config and request.

    Inference uses lerobot-record with --policy.path and NO --teleop.* flags.
    The policy replaces the teleoperator and controls the robot autonomously.
    """
    cameras_dict = {}
    if config.mode == "bimanual":
        for cam in config.bimanual.cameras:
            cameras_dict[cam.name] = {
                "type": "opencv",
                "index_or_path": cam.index,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }

        bi = config.bimanual
        return [
            "lerobot-record",
            "--robot.type=bi_so101_follower",
            f"--robot.left_arm_port={bi.left_follower_port}",
            f"--robot.right_arm_port={bi.right_follower_port}",
            f"--robot.id={bi.follower_id or 'bimanual_follower'}",
            f"--robot.cameras={json.dumps(cameras_dict)}",
            f"--dataset.repo_id={request.repo_id}",
            f"--dataset.single_task={request.single_task}",
            f"--dataset.num_episodes={request.num_episodes}",
            f"--dataset.episode_time_s={request.episode_time_s}",
            f"--display_data={str(request.display_data).lower()}",
            f"--policy.path={request.policy_path}",
        ]
    else:
        for cam in config.single_arm.cameras:
            cameras_dict[cam.name] = {
                "type": "opencv",
                "index_or_path": cam.index,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
            }

        sa = config.single_arm
        return [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={sa.follower_port}",
            f"--robot.id={sa.follower_id or 'single_follower'}",
            f"--robot.cameras={json.dumps(cameras_dict)}",
            f"--dataset.repo_id={request.repo_id}",
            f"--dataset.single_task={request.single_task}",
            f"--dataset.num_episodes={request.num_episodes}",
            f"--dataset.episode_time_s={request.episode_time_s}",
            f"--display_data={str(request.display_data).lower()}",
            f"--policy.path={request.policy_path}",
        ]


def _extract_inference_ports(config) -> list[str]:
    """Extract follower ports used by inference (no teleop ports needed)."""
    if config.mode == "bimanual":
        bi = config.bimanual
        return [p for p in [bi.left_follower_port, bi.right_follower_port] if p]
    else:
        return [config.single_arm.follower_port] if config.single_arm.follower_port else []


@router.post("/start", response_model=InferenceResponse)
async def start_inference(request: InferenceRequest):
    """Start policy inference (autonomous robot control)."""
    ports = []
    try:
        config = config_manager.load_config()

        # Validate config - only need robot ports and cameras (no teleop needed)
        if config.mode == "bimanual":
            if not all(
                [
                    config.bimanual.left_follower_port,
                    config.bimanual.right_follower_port,
                ]
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Bimanual mode requires both follower arm ports to be configured",
                )

            if not config.bimanual.cameras:
                raise HTTPException(
                    status_code=400, detail="No cameras configured for inference"
                )
        else:
            if not config.single_arm.follower_port:
                raise HTTPException(
                    status_code=400,
                    detail="Single arm mode requires a follower port to be configured",
                )

            if not config.single_arm.cameras:
                raise HTTPException(
                    status_code=400, detail="No cameras configured for inference"
                )

        # Acquire port locks
        ports = _extract_inference_ports(config)
        try:
            await port_lock_manager.acquire(ports, owner="inference", mode="subprocess")
        except PortInUseError as e:
            raise HTTPException(status_code=409, detail={"message": str(e), "owner": e.owner, "port": e.port})

        # Clear stale eval dataset cache to prevent conflicts on re-runs
        cache_cleared = False
        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / request.repo_id
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_cleared = True
            logger.info("Cleared stale eval cache at %s", cache_dir)

        command = build_inference_command(config, request)
        process_id = await process_manager.start_process(command, "inference")

        # Register process→ports mapping for release on stop
        await port_lock_manager.register_process(process_id, ports)

        msg = "Inference started successfully"
        if cache_cleared:
            msg += " (previous eval cache cleared)"

        return InferenceResponse(process_id=process_id, message=msg)

    except HTTPException:
        raise
    except Exception as e:
        if ports:
            await port_lock_manager.release(ports)
        raise HTTPException(status_code=500, detail=f"Failed to start inference: {e}")


@router.post("/stop/{process_id}")
async def stop_inference(process_id: str):
    """Stop inference."""
    try:
        success = await process_manager.stop_process(process_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Process {process_id} not found"
            )

        # Wait for OS to release ports, then release locks
        await asyncio.sleep(0.5)
        await port_lock_manager.release_for_process(process_id)

        return {"message": "Inference stopped successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop inference: {e}")


@router.get("/status/{process_id}", response_model=ProcessStatus)
async def get_inference_status(process_id: str):
    """Get inference process status."""
    try:
        status = await process_manager.get_status(process_id)

        if not status:
            raise HTTPException(
                status_code=404, detail=f"Process {process_id} not found"
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")


# --- MolmoAct2 (in-process; same router as policy inference) -------------------


@router.post("/molmo/load")
async def inference_molmo_load(body: MolmoLoadRequest):
    """Load MolmoAct2 weights and run CUDA graph warmup (background thread)."""
    if body.remote_host and str(body.remote_host).strip():
        cfg = config_manager.load_config()
        cfg.remote_inference_host = body.remote_host.strip()
        config_manager.save_config(cfg)
        raise HTTPException(
            status_code=501,
            detail={
                "error": "remote_inference_host is stored but remote GPU worker is not implemented yet; "
                "run the backend on the GPU machine with an empty remote_host, or leave remote_host blank for local load."
            },
        )

    cfg = config_manager.load_config()
    cfg.remote_inference_host = None
    config_manager.save_config(cfg)

    eng = get_molmo_engine()
    if eng is not None and eng.is_ready():
        return {"status": "already_loaded", "message": "MolmoAct2 model is already loaded and ready."}

    tag = start_background_load(DEFAULT_MOLMO_MODEL_ID, body.device)
    if tag == "already_loaded":
        return {"status": "already_loaded", "message": "MolmoAct2 model is already loaded and ready."}
    if tag == "loading_in_progress":
        return {"status": "loading", "message": "Load already in progress."}
    return {"status": "loading"}


@router.get("/molmo/status")
async def inference_molmo_status():
    """MolmoAct2 load / warmup progress."""
    eng = get_molmo_engine()
    if eng is None:
        return {
            "loaded": False,
            "ready": False,
            "device": "",
            "warmup_progress": 0,
        }
    return {
        "loaded": eng.loaded,
        "ready": eng.is_ready(),
        "device": eng.device,
        "warmup_progress": eng.warmup_progress,
        "load_error": eng.load_error,
    }


@router.post("/molmo/start")
async def inference_molmo_start(body: MolmoStartRequest):
    """Start MolmoAct2 perception–action loop."""
    eng = get_molmo_engine()
    if eng is None or not eng.is_ready():
        _molmo_not_loaded()

    cur = get_inference_loop()
    if cur is not None:
        st = cur.get_status()
        if st.get("running"):
            raise HTTPException(status_code=409, detail={"error": "inference loop already running"})

    if cur is not None:
        cur.stop()

    loop = InferenceLoop(
        eng,
        camera_indices=body.camera_indices,
        hz=body.hz,
        robot_port=body.robot_port if not body.dry_run else "",
        dry_run=body.dry_run,
    )
    set_inference_loop(loop)
    ok = loop.start(body.task)
    if not ok:
        raise HTTPException(status_code=500, detail={"error": "failed to start loop"})
    return {"status": "started"}


@router.post("/molmo/stop")
async def inference_molmo_stop():
    """Stop MolmoAct2 control loop (does not unload weights)."""
    loop = get_inference_loop()
    if loop is not None:
        loop.stop()
    return {"status": "stopped"}


@router.get("/molmo/loop_status")
async def inference_molmo_loop_status():
    """Live step / latency / last action for MolmoAct2 loop."""
    eng = get_molmo_engine()
    if eng is None or not eng.is_ready():
        _molmo_not_loaded()
    loop = get_inference_loop()
    if loop is None:
        return {
            "running": False,
            "step_count": 0,
            "last_action": None,
            "last_latency_ms": 0.0,
            "task": "",
            "action_dim": 0,
        }
    return loop.get_status()


# --- MolmoAct2 (in-process; same router as policy inference) -------------------


@router.post("/molmo/load")
async def inference_molmo_load(body: MolmoLoadRequest):
    """Load MolmoAct2 weights and run CUDA graph warmup (background thread)."""
    if body.remote_host and str(body.remote_host).strip():
        cfg = config_manager.load_config()
        cfg.remote_inference_host = body.remote_host.strip()
        config_manager.save_config(cfg)
        raise HTTPException(
            status_code=501,
            detail={
                "error": "remote_inference_host is stored but remote GPU worker is not implemented yet; "
                "run the backend on the GPU machine with an empty remote_host, or leave remote_host blank for local load."
            },
        )

    cfg = config_manager.load_config()
    cfg.remote_inference_host = None
    config_manager.save_config(cfg)

    eng = get_molmo_engine()
    if eng is not None and eng.is_ready():
        return {"status": "already_loaded", "message": "MolmoAct2 model is already loaded and ready."}

    tag = start_background_load(DEFAULT_MOLMO_MODEL_ID, body.device)
    if tag == "already_loaded":
        return {"status": "already_loaded", "message": "MolmoAct2 model is already loaded and ready."}
    if tag == "loading_in_progress":
        return {"status": "loading", "message": "Load already in progress."}
    return {"status": "loading"}


@router.get("/molmo/status")
async def inference_molmo_status():
    """MolmoAct2 load / warmup progress."""
    eng = get_molmo_engine()
    if eng is None:
        return {
            "loaded": False,
            "ready": False,
            "device": "",
            "warmup_progress": 0,
        }
    return {
        "loaded": eng.loaded,
        "ready": eng.is_ready(),
        "device": eng.device,
        "warmup_progress": eng.warmup_progress,
        "load_error": eng.load_error,
    }


@router.post("/molmo/start")
async def inference_molmo_start(body: MolmoStartRequest):
    """Start MolmoAct2 perception–action loop."""
    eng = get_molmo_engine()
    if eng is None or not eng.is_ready():
        _molmo_not_loaded()

    cur = get_inference_loop()
    if cur is not None:
        st = cur.get_status()
        if st.get("running"):
            raise HTTPException(status_code=409, detail={"error": "inference loop already running"})

    if cur is not None:
        cur.stop()

    loop = InferenceLoop(
        eng,
        camera_indices=body.camera_indices,
        hz=body.hz,
        robot_port=body.robot_port if not body.dry_run else "",
        dry_run=body.dry_run,
    )
    set_inference_loop(loop)
    ok = loop.start(body.task)
    if not ok:
        raise HTTPException(status_code=500, detail={"error": "failed to start loop"})
    return {"status": "started"}


@router.post("/molmo/stop")
async def inference_molmo_stop():
    """Stop MolmoAct2 control loop (does not unload weights)."""
    loop = get_inference_loop()
    if loop is not None:
        loop.stop()
    return {"status": "stopped"}


@router.get("/molmo/loop_status")
async def inference_molmo_loop_status():
    """Live step / latency / last action for MolmoAct2 loop."""
    eng = get_molmo_engine()
    if eng is None or not eng.is_ready():
        _molmo_not_loaded()
    loop = get_inference_loop()
    if loop is None:
        return {
            "running": False,
            "step_count": 0,
            "last_action": None,
            "last_latency_ms": 0.0,
            "task": "",
            "action_dim": 0,
        }
    return loop.get_status()
