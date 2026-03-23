"""REST API endpoints for base (LeKiwi) motor control."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.base_control import base_control_service

logger = logging.getLogger(__name__)
router = APIRouter()


class DetectRequest(BaseModel):
    ports: list[str]


class DetectResponse(BaseModel):
    detected_port: str | None
    message: str


class ConnectRequest(BaseModel):
    port: str


class StatusResponse(BaseModel):
    connected: bool
    port: str | None
    speed_index: int


@router.post("/detect", response_model=DetectResponse)
async def detect_base(req: DetectRequest):
    """Auto-detect which port has base motors (IDs 7, 8, 9)."""
    port = base_control_service.detect_base_port(req.ports)
    if port:
        return DetectResponse(detected_port=port, message=f"Base motors found on {port}")
    return DetectResponse(detected_port=None, message="No base motors detected on any port")


@router.post("/connect", response_model=StatusResponse)
async def connect_base(req: ConnectRequest):
    """Connect to base motors on the specified port."""
    try:
        base_control_service.connect(req.port)
        return StatusResponse(
            connected=True, port=req.port, speed_index=base_control_service.speed_index,
        )
    except Exception as e:
        logger.error(f"Failed to connect base on {req.port}: {e}")
        raise


@router.post("/disconnect", response_model=StatusResponse)
async def disconnect_base():
    """Disconnect from base motors and stop wheels."""
    base_control_service.disconnect()
    return StatusResponse(connected=False, port=None, speed_index=0)


@router.get("/status", response_model=StatusResponse)
async def base_status():
    """Get current base control status."""
    return StatusResponse(
        connected=base_control_service.is_connected,
        port=base_control_service.port,
        speed_index=base_control_service.speed_index,
    )
