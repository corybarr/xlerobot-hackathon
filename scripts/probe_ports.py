"""Probe every visible serial port for an SO-101 feetech board.

For each COM/tty port, attempt to open a FeetechMotorsBus with the 6 SO-101
motor IDs and read present_position. Reports which ports respond and the
current encoder values. Helps map COM port -> physical arm without moving
anything (read-only).

Usage:
    python scripts/probe_ports.py
    python scripts/probe_ports.py --port COM7 --port COM9   # subset
"""
from __future__ import annotations

import argparse
import sys

from serial.tools import list_ports

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}


def _make_motors() -> dict:
    motors = {}
    for name in MOTOR_NAMES:
        norm = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(MOTOR_IDS[name], "sts3215", norm)
    return motors


def probe_port(port: str) -> dict:
    """Open the bus, read positions, disconnect. No torque/movement changes."""
    result: dict = {"port": port, "ok": False, "error": None, "positions": None}
    bus = None
    try:
        bus = FeetechMotorsBus(port=port, motors=_make_motors())
        bus.connect()
        positions = bus.sync_read("Present_Position", MOTOR_NAMES, normalize=False)
        result["ok"] = True
        result["positions"] = positions
    except Exception as e:
        msg = str(e).strip().splitlines()[0] if str(e).strip() else "unknown"
        result["error"] = f"{type(e).__name__}: {msg[:160]}"
    finally:
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass
    return result


def discover_ports() -> list[tuple[str, str, str]]:
    """Return [(device, description, hwid)] for every port the OS sees."""
    return [(p.device, p.description or "?", p.hwid or "") for p in list_ports.comports()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", action="append", help="Specific port to probe (repeatable). Default: all visible ports.")
    args = ap.parse_args()

    discovered = discover_ports()
    if args.port:
        discovered = [d for d in discovered if d[0] in set(args.port)]

    if not discovered:
        print("No serial ports visible.", file=sys.stderr)
        return 1

    print(f"Probing {len(discovered)} port(s):\n")
    results = []
    for device, desc, hwid in discovered:
        print(f"=== {device}  ({desc})")
        if hwid:
            print(f"    hwid: {hwid}")
        r = probe_port(device)
        if r["ok"]:
            print(f"    OK -- 6/6 motors responded.")
            print(f"    positions: {r['positions']}")
        else:
            print(f"    FAIL -- {r['error']}")
        results.append(r)
        print()

    alive = [r for r in results if r["ok"]]
    print(f"Summary: {len(alive)}/{len(results)} port(s) have responsive SO-101 boards.")
    return 0 if alive else 1


if __name__ == "__main__":
    sys.exit(main())
