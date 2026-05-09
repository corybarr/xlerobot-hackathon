# Model: allenai/MolmoAct2-SO100_101 — https://huggingface.co/allenai/MolmoAct2-SO100_101
"""In-process MolmoAct2 inference engine and control loop.

Bimanual setups may require a larger state vector than single-arm (6,); the
checkpoint uses ``norm_tag="so100_so101_molmoact2"``. We read ``norm_stats.json``
from the Hub when available; otherwise default ``state_dim`` is 6.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MOLMO_MODEL_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

# --- log broadcast (thread-safe, fan-out to asyncio subscribers) ----------------

class MolmoLogHub:
    """Recent lines + live subscribers for WebSocket streaming."""

    def __init__(self, maxlen: int = 500):
        self._recent: deque[str] = deque(maxlen=maxlen)
        self._recent_lock = threading.Lock()
        self._subs: List[asyncio.Queue] = []
        self._subs_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def append(self, line: str) -> None:
        with self._recent_lock:
            self._recent.append(line)
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        def _fan_out() -> None:
            with self._subs_lock:
                qs = list(self._subs)
            for q in qs:
                try:
                    q.put_nowait(line)
                except asyncio.QueueFull:
                    pass
                except Exception:
                    pass

        try:
            loop.call_soon_threadsafe(_fan_out)
        except RuntimeError:
            pass

    def snapshot(self) -> List[str]:
        with self._recent_lock:
            return list(self._recent)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._subs_lock:
            self._subs.append(q)
        for line in self.snapshot():
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                break
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)


_molmo_log_hub = MolmoLogHub(maxlen=500)


def molmo_append_log(line: str) -> None:
    _molmo_log_hub.append(line)


def get_molmo_log_hub() -> MolmoLogHub:
    return _molmo_log_hub


def _resolve_state_dim(model_id: str) -> int:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=model_id, filename="norm_stats.json", repo_type="model")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        stats = data.get("norm_stats") or data
        if isinstance(stats, dict):
            st = stats.get("state")
            if isinstance(st, dict) and "mean" in st:
                mean = st["mean"]
                if isinstance(mean, list):
                    return len(mean)
    except Exception as e:
        logger.info("Could not resolve state_dim from Hub (%s); using 6", e)
    return 6


def capture_frames(camera_indices: List[int]) -> List[Image.Image]:
    """OpenCV capture: one RGB PIL image per camera index."""
    import cv2

    out: List[Image.Image] = []
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        try:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(rgb))
        finally:
            cap.release()
    return out


class _CameraSession:
    """Reuse ``VideoCapture`` handles across control steps."""

    def __init__(self, camera_indices: List[int]):
        import cv2

        self._cv2 = cv2
        self._indices = camera_indices
        self._caps = [cv2.VideoCapture(i) for i in camera_indices]

    def read_rgb_pil(self) -> List[Image.Image]:
        out: List[Image.Image] = []
        for cap in self._caps:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(rgb))
        return out

    def release(self) -> None:
        for cap in self._caps:
            cap.release()
        self._caps.clear()


# --- engine -------------------------------------------------------------------


class MolmoAct2InferenceEngine:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.ready = False
        self.loaded = False
        self.warmup_progress = 0
        self.state_dim = _resolve_state_dim(model_id)
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()

    def is_ready(self) -> bool:
        return bool(self.ready and self.model is not None and self.processor is not None)

    def warmup(self, n: int = 5) -> None:
        import torch
        from PIL import Image as PILImage

        if self.model is None or self.processor is None:
            return
        use_graph = self.device == "cuda" and torch.cuda.is_available()
        dummy = PILImage.new("RGB", (640, 480))
        state = np.zeros(self.state_dim, dtype=np.float32)
        self.warmup_progress = 0
        molmo_append_log("MolmoAct2: starting CUDA graph warmup...")
        for i in range(n):
            _ = self.model.predict_action(
                processor=self.processor,
                images=[dummy, dummy],
                task="warm up",
                state=state,
                norm_tag=NORM_TAG,
                num_steps=10,
                enable_cuda_graph=use_graph,
            )
            self.warmup_progress = i + 1
            molmo_append_log(f"MolmoAct2: warmup {self.warmup_progress}/{n}")
        self.ready = True
        molmo_append_log("MolmoAct2: warmup complete; ready for inference.")

    def predict(self, images: List[Image.Image], task: str, state: np.ndarray) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError("Model is not ready")
        use_graph = self.device == "cuda" and torch.cuda.is_available()
        st = np.asarray(state, dtype=np.float32).flatten()
        if st.size < self.state_dim:
            st = np.pad(st, (0, self.state_dim - st.size))
        elif st.size > self.state_dim:
            st = st[: self.state_dim]
        out = self.model.predict_action(
            processor=self.processor,
            images=images,
            task=task,
            state=st,
            norm_tag=NORM_TAG,
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=10,
            normalize_language=True,
            enable_cuda_graph=use_graph,
        )
        actions = out.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        return np.asarray(actions, dtype=np.float32)


_engine: Optional[MolmoAct2InferenceEngine] = None
_engine_singleton_lock = threading.Lock()


def get_molmo_engine() -> Optional[MolmoAct2InferenceEngine]:
    return _engine


def _load_engine_worker(model_id: str, device: str) -> None:
    global _engine
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    eng = MolmoAct2InferenceEngine(model_id, device)
    with _engine_singleton_lock:
        _engine = eng
    try:
        molmo_append_log(f"MolmoAct2: loading {model_id} on {device}...")
        eng.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float32
        m = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        if device == "cuda" and torch.cuda.is_available():
            m = m.to("cuda")
        else:
            m = m.to("cpu")
        m.eval()
        eng.model = m
        eng.loaded = True
        eng.load_error = None
        molmo_append_log("MolmoAct2: weights loaded; running warmup.")
        eng.warmup(5)
        molmo_append_log("MolmoAct2: engine ready.")
    except Exception as e:
        logger.exception("MolmoAct2 load failed")
        eng.loaded = False
        eng.ready = False
        eng.load_error = str(e)
        eng.model = None
        eng.processor = None
        molmo_append_log(f"ERROR: MolmoAct2 load failed: {e}")


_load_thread: Optional[threading.Thread] = None
_load_thread_lock = threading.Lock()


def start_background_load(model_id: str, device: str) -> str:
    """Spawn thread to load model + warmup. Returns status tag for HTTP layer."""

    global _load_thread
    eng = get_molmo_engine()
    if eng is not None and eng.is_ready():
        return "already_loaded"
    with _load_thread_lock:
        if _load_thread is not None and _load_thread.is_alive():
            return "loading_in_progress"

        def run() -> None:
            global _load_thread
            try:
                _load_engine_worker(model_id, device)
            finally:
                with _load_thread_lock:
                    _load_thread = None

        _load_thread = threading.Thread(target=run, daemon=True)
        _load_thread.start()
    return "loading"


# --- inference loop -----------------------------------------------------------


class InferenceLoop:
    def __init__(
        self,
        engine: MolmoAct2InferenceEngine,
        camera_indices: List[int],
        hz: float,
        robot_port: str,
        dry_run: bool,
    ):
        self.engine = engine
        self.camera_indices = camera_indices
        self.hz = max(0.5, min(10.0, hz))
        self.robot_port = (robot_port or "").strip()
        self.dry_run = dry_run
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_lock = threading.Lock()
        self.step_count = 0
        self.last_action: Optional[List[float]] = None
        self.last_latency_ms = 0.0
        self.task = ""
        self._cam: Optional[_CameraSession] = None
        self._arm: Optional[Any] = None

    def start(self, task: str) -> bool:
        with self._run_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if not self.engine.is_ready():
                return False
            self.task = task
            self._stop.clear()
            self.step_count = 0
            self.last_action = None
            self.last_latency_ms = 0.0
            self._cam = _CameraSession(self.camera_indices)
            if not self.dry_run and self.robot_port:
                from backend.molmo_robot_bridge import MolmoArmSession

                self._arm = MolmoArmSession(self.robot_port)
                self._arm.connect()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            molmo_append_log(f"Inference loop started (hz={self.hz}, dry_run={self.dry_run}).")
            return True

    def _run_loop(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                assert self._cam is not None
                images = self._cam.read_rgb_pil()
                if len(images) < 2:
                    while len(images) < 2:
                        images.append(Image.new("RGB", (640, 480)))

                if self._arm is not None:
                    state = self._arm.read_state_normalized()
                else:
                    state = np.zeros(self.engine.state_dim, dtype=np.float32)

                if state.size != self.engine.state_dim:
                    s = np.zeros(self.engine.state_dim, dtype=np.float32)
                    n = min(state.size, self.engine.state_dim)
                    s[:n] = state.flatten()[:n]
                    state = s

                actions = self.engine.predict(images, self.task, state)
                flat = np.asarray(actions, dtype=np.float32).flatten()

                if self._arm is not None:
                    self._arm.send_normalized_positions(flat)
                else:
                    molmo_append_log(f"[dry-run] mock command: {flat.tolist()}")

                self.step_count += 1
                self.last_action = [float(x) for x in flat.tolist()]
                self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
                molmo_append_log(
                    f"step={self.step_count} latency_ms={self.last_latency_ms:.1f} action={self.last_action}"
                )
            except Exception as e:
                logger.exception("inference step error")
                molmo_append_log(f"ERROR step: {e}")
            finally:
                if self._stop.wait(timeout=period):
                    break
        self._cleanup_hardware()
        molmo_append_log("Inference loop stopped.")

    def _cleanup_hardware(self) -> None:
        if self._cam is not None:
            self._cam.release()
            self._cam = None
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        self._thread = None
        self._cleanup_hardware()

    def get_status(self) -> Dict[str, Any]:
        running = self._thread is not None and self._thread.is_alive()
        return {
            "running": running,
            "step_count": self.step_count,
            "last_action": self.last_action,
            "last_latency_ms": self.last_latency_ms,
            "task": self.task,
            "action_dim": len(self.last_action) if self.last_action else 0,
        }


_loop_holder: Optional[InferenceLoop] = None
_loop_mutex = threading.Lock()


def get_inference_loop() -> Optional[InferenceLoop]:
    return _loop_holder


def set_inference_loop(loop: Optional[InferenceLoop]) -> None:
    global _loop_holder
    _loop_holder = loop
