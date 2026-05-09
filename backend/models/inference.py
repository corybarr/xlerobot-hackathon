"""Inference-related models."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Inference start request."""

    policy_path: str = Field(..., description="Path to the trained policy (local folder or HF repo)")
    repo_id: str = Field(..., description="HuggingFace repo ID for evaluation results (username/eval_dataset)")
    single_task: str = Field(..., description="Task description (should match training task)")
    num_episodes: int = Field(10, description="Number of evaluation episodes")
    episode_time_s: int = Field(50, description="Episode duration in seconds")
    display_data: bool = Field(True, description="Whether to show visualization")


class InferenceResponse(BaseModel):
    """Inference start response."""

    process_id: str = Field(..., description="Process identifier for tracking")
    message: str = Field(..., description="Status message")


class MolmoLoadRequest(BaseModel):
    """MolmoAct2 model load + CUDA warmup."""

    device: Literal["cuda", "cpu"] = "cuda"
    remote_host: Optional[str] = Field(
        None, description="Reserved; remote GPU worker not implemented in this phase"
    )


class MolmoStartRequest(BaseModel):
    """Start in-process MolmoAct2 control loop."""

    task: str = Field(..., min_length=1)
    camera_indices: List[int] = Field(..., min_length=2, max_length=2)
    robot_port: str = Field(default="/dev/ttyACM0")
    hz: float = Field(default=2.0, ge=0.5, le=10.0)
    dry_run: bool = False
