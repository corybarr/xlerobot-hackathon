"""SO101 follower bridge for MolmoAct2 inference (Feetech bus via LeRobot).

This module imports LeRobot motor code; keep ``backend.inference`` free of
``lerobot`` imports per project constraints.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from backend.services.manual_calibration import MOTOR_NAMES, MOTOR_IDS

logger = logging.getLogger(__name__)


def _create_control_bus(port: str) -> FeetechMotorsBus:
    """Connect bus suitable for position control (torque will be enabled separately)."""
    motors = {}
    for name in MOTOR_NAMES:
        norm_mode = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(MOTOR_IDS[name], "sts3215", norm_mode)
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    for motor in MOTOR_NAMES:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", motor, 16)
        bus.write("I_Coefficient", motor, 0)
        bus.write("D_Coefficient", motor, 32)
        if motor == "gripper":
            bus.write("Max_Torque_Limit", motor, 500)
            bus.write("Protection_Current", motor, 250)
            bus.write("Overload_Torque", motor, 25)
    bus.enable_torque()
    return bus


class MolmoArmSession:
    """Single-arm SO101 session: read normalized state, send normalized goal positions."""

    def __init__(self, port: str):
        self.port = port
        self._bus: Optional[FeetechMotorsBus] = None

    def connect(self) -> None:
        if self._bus is not None:
            return
        self._bus = _create_control_bus(self.port)

    def disconnect(self) -> None:
        if self._bus is None:
            return
        try:
            if self._bus.is_connected:
                self._bus.disconnect(disable_torque=True)
        except Exception as e:
            logger.warning("MolmoArmSession disconnect: %s", e)
        finally:
            self._bus = None

    def read_state_normalized(self) -> np.ndarray:
        if self._bus is None:
            raise RuntimeError("Bus not connected")
        pos = self._bus.sync_read("Present_Position", list(MOTOR_NAMES))
        return np.asarray([float(pos[m]) for m in MOTOR_NAMES], dtype=np.float32)

    def send_normalized_positions(self, actions: np.ndarray) -> None:
        if self._bus is None:
            raise RuntimeError("Bus not connected")
        n = min(len(actions), len(MOTOR_NAMES))
        goal = {MOTOR_NAMES[i]: float(actions[i]) for i in range(n)}
        self._bus.sync_write("Goal_Position", goal)


def send_joint_commands(port: str, actions: np.ndarray, dry_run: bool) -> None:
    """Apply joint targets on one follower arm, or log-only in dry-run mode.

    ``actions`` is length 6 in motor order (shoulder_pan … gripper), normalized
    to LeRobot / Feetech conventions.
    """
    flat = np.asarray(actions, dtype=np.float32).flatten()
    if dry_run:
        logger.info("[dry-run] mock joint command port=%s values=%s", port, flat.tolist())
        return
    if not port or not port.strip():
        logger.warning("[dry-run fallback] empty robot port; logging commands: %s", flat.tolist())
        return
    session = MolmoArmSession(port.strip())
    try:
        session.connect()
        session.send_normalized_positions(flat)
    finally:
        session.disconnect()
