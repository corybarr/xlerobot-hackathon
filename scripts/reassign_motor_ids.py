"""Reassign Feetech motor IDs in-place over the bus.

When motors on an SO-101 arm have non-standard IDs (e.g. 2,4,5,6,7,8 instead
of 1-6), we can rewrite each motor's ID register over the bus without
disconnecting any cables. Order matters: we walk current IDs ascending and
write target IDs ascending (1,2,3,...), so each step never collides with a
still-active higher ID.

ASSUMPTION: the motors were physically assembled in standard SO-101 order
(base -> gripper). i.e. the lowest current ID is the shoulder_pan, the
highest is the gripper. If your arm was assembled out of order, this will
re-ID the wrong motor as wrong joint and motion will be scrambled.

Usage:
    python scripts/reassign_motor_ids.py --port COM10 --initial-ids 2,4,5,6,7,8
"""
from __future__ import annotations

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _make_motors_at_target_ids() -> dict:
    """Bus configured with TARGET IDs (1..6 = standard SO-101)."""
    motors = {}
    for i, name in enumerate(MOTOR_NAMES):
        norm = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(i + 1, "sts3215", norm)
    return motors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--initial-ids", required=True,
                   help="Current motor IDs in joint order (base->gripper), comma-separated. e.g. 2,4,5,6,7,8")
    p.add_argument("--baud", type=int, default=1_000_000)
    args = p.parse_args()

    initial_ids = [int(x) for x in args.initial_ids.split(",")]
    if len(initial_ids) != 6:
        print(f"ERROR: need exactly 6 IDs, got {len(initial_ids)}", file=sys.stderr)
        return 1

    # Sort so we always write the lowest target ID first
    pairs = list(zip(MOTOR_NAMES, initial_ids))
    pairs.sort(key=lambda x: x[1])  # ascending by current ID

    print(f"Reassign plan on {args.port}:")
    for i, (name, old_id) in enumerate(pairs):
        new_id = i + 1
        marker = "(already correct)" if old_id == new_id else f"({old_id} -> {new_id})"
        print(f"  {name:14s}  current ID {old_id}  =>  new ID {new_id}  {marker}")
    print()

    bus = FeetechMotorsBus(port=args.port, motors=_make_motors_at_target_ids())

    for i, (name, old_id) in enumerate(pairs):
        new_id = i + 1
        if old_id == new_id:
            print(f"  {name}: skipping (already at ID {new_id})")
            continue
        print(f"  {name}: writing ID {old_id} -> {new_id} ...", end=" ", flush=True)
        try:
            bus.setup_motor(name, initial_baudrate=args.baud, initial_id=old_id)
            print("OK")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:160]}")
            try:
                bus.disconnect()
            except Exception:
                pass
            return 1

    try:
        bus.disconnect()
    except Exception:
        pass

    print("\nReassignment complete. Verifying with a fresh connect at standard IDs...")
    bus2 = FeetechMotorsBus(port=args.port, motors=_make_motors_at_target_ids())
    try:
        bus2.connect()
        positions = bus2.sync_read("Present_Position", MOTOR_NAMES, normalize=False)
        print(f"  All 6 motors at standard IDs respond. Positions: {positions}")
        bus2.disconnect()
        return 0
    except Exception as e:
        print(f"  Verification FAILED: {type(e).__name__}: {str(e)[:200]}")
        try:
            bus2.disconnect()
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    sys.exit(main())
