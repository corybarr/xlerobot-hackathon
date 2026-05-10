#!/usr/bin/env python3
"""mm — CLI wrapper for the MakerMods-App REST API.

Drives teleop / calibration / recording / training / inference from terminal,
without clicking through the wizard. Mirrors the UI's behavior including its
port-lock manager and live log streaming over WebSocket.

Subcommands:
  health                                — backend healthcheck
  ports                                 — list serial ports
  cameras                               — list cameras
  cameras preview [INDICES...]          — capture preview JPEGs
  wiggle <port>                         — wiggle the gripper at <port>

  calibrate status                      — current cal state per device
  calibrate missing                     — devices missing cal
  calibrate auto <port> <device_id>     — full auto-cal w/ live log stream
                                          --type so101_follower (default)
                                          --no-complete to skip post-cal copy
  calibrate stop <process_id>

  config show                           — current webui_config.json
  config set KEY=VAL [KEY=VAL...]       — set top-level config keys (JSON values)

  record start KEY=VAL ...              — POST /api/recording/start (k=v body)
  record status <process_id>
  record stop <process_id>
  record cache clear

  teleop start KEY=VAL ...
  teleop stop <process_id>

  train start KEY=VAL ...
  train status <job_id>
  train cancel <job_id>

  hf whoami
  hf repos

  processes                             — backend system status
  locks                                 — currently held port locks

Env:
  MM_HOST     default http://localhost:8000

Examples:
  ./scripts/mm.py health
  ./scripts/mm.py ports
  ./scripts/mm.py wiggle COM10
  ./scripts/mm.py calibrate auto COM10 right_follower
  ./scripts/mm.py record start \\
    robot.type=so101_follower robot.port=COM10 robot.id=right_follower \\
    teleop.type=so101_leader  teleop.port=COM7  teleop.id=right_leader \\
    dataset.repo_id=Globalmysterysnailrevolution/xlerobot-place-fork \\
    dataset.single_task='Place the fork' dataset.num_episodes=10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Optional

import requests

MM_HOST = os.getenv("MM_HOST", "http://localhost:8000")

# ANSI colors (no extra deps)
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"; DIM = "\033[2m"; RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _url(path: str) -> str:
    return f"{MM_HOST.rstrip('/')}{path}"


def _get(path: str, params: Optional[dict] = None, timeout: int = 30) -> Any:
    r = requests.get(_url(path), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.content else None


def _post(path: str, body: Optional[dict] = None, timeout: int = 60) -> Any:
    r = requests.post(_url(path), json=body or {}, timeout=timeout)
    if not r.ok:
        try:
            err = r.json()
            print(f"{RED}HTTP {r.status_code}: {json.dumps(err)}{RESET}", file=sys.stderr)
        except Exception:
            print(f"{RED}HTTP {r.status_code}: {r.text[:500]}{RESET}", file=sys.stderr)
        sys.exit(2)
    return r.json() if r.content else None


def _delete(path: str, params: Optional[dict] = None, timeout: int = 30) -> Any:
    r = requests.delete(_url(path), params=params, timeout=timeout)
    r.raise_for_status()
    return r.json() if r.content else None


def _kv_to_body(pairs: list[str]) -> dict:
    """Turn ['k=v', 'k2=v2'] into {'k': v, 'k2': v2}, parsing JSON-ish values."""
    body: dict = {}
    for p in pairs:
        if "=" not in p:
            print(f"{RED}bad arg: {p!r} — expected key=value{RESET}", file=sys.stderr)
            sys.exit(2)
        k, _, v = p.partition("=")
        # try int, float, bool, json — fall back to string
        if v.lower() == "true":
            parsed: Any = True
        elif v.lower() == "false":
            parsed = False
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    if v.startswith("{") or v.startswith("["):
                        try:
                            parsed = json.loads(v)
                        except json.JSONDecodeError:
                            parsed = v
                    else:
                        parsed = v
        body[k.strip()] = parsed
    return body


def _pretty(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# WebSocket log stream
# ---------------------------------------------------------------------------

async def _stream_logs(process_id: str) -> None:
    try:
        import websockets
    except ImportError:
        print(f"{YELLOW}note: websockets package missing; skipping live log stream.{RESET}", file=sys.stderr)
        return
    ws_url = MM_HOST.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + f"/ws/logs/{process_id}"
    print(f"{DIM}--- streaming logs from {ws_url} ---{RESET}")
    try:
        async with websockets.connect(ws_url, ping_interval=20) as ws:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    line = msg.get("line", str(msg))
                    level = (msg.get("level") or "").lower()
                    color = RED if level == "error" else (YELLOW if level == "warn" else "")
                    print(f"  {color}{line}{RESET if color else ''}")
                except json.JSONDecodeError:
                    print(f"  {raw}")
    except Exception as e:
        # Connection closes when process exits — that's normal.
        if "1000" in str(e) or "closed" in str(e).lower():
            print(f"{DIM}--- log stream closed (process exited) ---{RESET}")
        else:
            print(f"{YELLOW}log stream error: {type(e).__name__}: {e}{RESET}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommands — setup
# ---------------------------------------------------------------------------

def cmd_health(args):
    _pretty(_get("/api/health", timeout=5))


def cmd_ports(args):
    ports = _get("/api/setup/ports")
    if not ports:
        print(f"{YELLOW}(no serial ports detected){RESET}")
        return
    for p in ports:
        hwid = p.get("hwid") or ""
        print(f"  {CYAN}{p['port']}{RESET}  {p.get('description','')}")
        if hwid:
            print(f"    {DIM}hwid: {hwid}{RESET}")


def cmd_cameras(args):
    if args.preview:
        indices = [int(i) for i in args.preview]
        body = indices if indices else None
        result = _post("/api/setup/cameras/preview", body=body)
        _pretty(result)
    else:
        cams = _get("/api/setup/cameras", timeout=60)
        if not cams:
            print(f"{YELLOW}(no cameras detected){RESET}")
            return
        for c in cams:
            print(f"  index {CYAN}{c['index']}{RESET}  {c['name']}  backend={c['backend']}  builtin={c['is_builtin']}")


def cmd_wiggle(args):
    print(f"  wiggling gripper on {args.port} ...")
    _post("/api/setup/wiggle", body={"port": args.port}, timeout=20)
    print(f"  {GREEN}done.{RESET} (the arm whose gripper just moved is at {args.port})")


# ---------------------------------------------------------------------------
# Subcommands — calibration (autocal is the priority)
# ---------------------------------------------------------------------------

def cmd_calibrate_status(args):
    _pretty(_get("/api/calibration/status"))


def cmd_calibrate_missing(args):
    _pretty(_get("/api/calibration/missing"))


def cmd_calibrate_auto(args):
    """Full auto-calibration flow: start → stream logs → complete."""
    print(f"{CYAN}auto-cal{RESET}  port={args.port}  device_id={args.device_id}  type={args.type}")
    start_resp = _post(
        "/api/calibration/auto/start",
        body={"port": args.port, "device_id": args.device_id},
    )
    process_id = start_resp["process_id"]
    print(f"  process_id: {process_id}")
    print(f"  {start_resp.get('message','')}")
    print()

    try:
        asyncio.run(_stream_logs(process_id))
    except KeyboardInterrupt:
        print(f"\n{YELLOW}interrupted — stopping process {process_id}...{RESET}")
        try:
            _post(f"/api/calibration/auto/stop/{process_id}")
        except Exception:
            pass
        sys.exit(130)

    if args.no_complete:
        print(f"{DIM}--no-complete: skipping post-cal copy.{RESET}")
        return

    print()
    print(f"  finalizing cal file (copy from so_follower/ to {args.type}/) ...")
    try:
        result = _post(
            f"/api/calibration/auto/complete/{args.device_id}",
            body={"category": "robots", "robot_type": args.type},
        )
        print(f"  {GREEN}done.{RESET}  saved at: {result.get('path')}")
    except SystemExit:
        # _post already printed the error
        pass


def cmd_calibrate_stop(args):
    _pretty(_post(f"/api/calibration/auto/stop/{args.process_id}"))


# ---------------------------------------------------------------------------
# Manual calibration via WebSocket (4-phase interactive flow)
# ---------------------------------------------------------------------------

async def _manual_cal_loop(port: str, device_id: str, device_type: str, robot_type: str) -> int:
    try:
        import websockets
    except ImportError:
        print(f"{RED}websockets package missing. install: pip install websockets{RESET}", file=sys.stderr)
        return 1

    ws_url = MM_HOST.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/api/calibration/manual/ws"
    print(f"{DIM}--- connecting to {ws_url} ---{RESET}")

    async with websockets.connect(ws_url, ping_interval=20, max_size=10_000_000) as ws:
        # Phase 1: open session
        await ws.send(json.dumps({
            "action": "start",
            "port": port,
            "device_type": device_type,
            "robot_type": robot_type,
            "device_id": device_id,
        }))
        msg = json.loads(await ws.recv())
        if msg.get("type") == "error":
            print(f"{RED}error: {msg.get('message')}{RESET}", file=sys.stderr)
            return 1
        motors = msg.get("motors", [])
        print(f"  {GREEN}connected.{RESET}  motors ({len(motors)}): {motors}")
        print()

        # Phase 2: set homing — user positions arm to middle
        print(f"  {CYAN}STEP 1{RESET}: Position the arm in its MIDDLE pose.")
        print(f"          Each joint centered. Arm extended naturally forward.")
        print(f"          The gripper jaws should be midway between fully open and fully closed.")
        await asyncio.to_thread(input, "          press Enter when ready: ")

        await ws.send(json.dumps({"action": "set_homing"}))
        msg = json.loads(await ws.recv())
        if msg.get("type") == "error":
            print(f"{RED}error setting homing: {msg.get('message')}{RESET}", file=sys.stderr)
            return 1
        offsets = msg.get("offsets", {})
        print(f"  {GREEN}homing offsets set{RESET} for {len(offsets)} motors")
        print()

        # Phase 3: range recording — user wiggles each joint through full range
        print(f"  {CYAN}STEP 2{RESET}: Move EACH joint through its full range of motion.")
        print(f"          Cover both directions for every joint. Open + close the gripper.")
        print(f"          Live pos + range below. Press Enter when done.")
        print()

        await ws.send(json.dumps({"action": "start_recording"}))
        # Acknowledge recording_started
        msg = json.loads(await ws.recv())
        if msg.get("type") != "recording_started":
            print(f"{YELLOW}unexpected msg waiting for recording_started: {msg}{RESET}")

        stop_event = asyncio.Event()
        latest: dict[str, dict] = {}

        async def watch_stdin():
            await asyncio.to_thread(input, "")
            stop_event.set()

        async def receive_positions():
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.3)
                    msg = json.loads(raw)
                    if msg.get("type") == "positions":
                        latest.update(msg.get("motors", {}))
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        async def display():
            first = True
            displayed_lines = 0
            while not stop_event.is_set():
                await asyncio.sleep(0.15)
                if not latest:
                    continue
                # Cursor: move up to overwrite previous block (in-place refresh)
                if not first:
                    sys.stdout.write(f"\033[{displayed_lines}A")
                first = False
                lines = []
                for name, m in latest.items():
                    pos = m.get("pos", 0)
                    mn = m.get("min", pos)
                    mx = m.get("max", pos)
                    span = mx - mn
                    bar_len = 32
                    if mx > mn:
                        pos_in_bar = max(0, min(bar_len - 1, int((pos - mn) / (mx - mn) * (bar_len - 1))))
                    else:
                        pos_in_bar = bar_len // 2
                    bar = "[" + " " * pos_in_bar + GREEN + "*" + RESET + " " * (bar_len - pos_in_bar - 1) + "]"
                    span_color = GREEN if span > 500 else (YELLOW if span > 100 else DIM)
                    lines.append(
                        f"\033[K  {name:14s} pos={pos:5d}  range=[{mn:5d},{mx:5d}]  "
                        f"{span_color}span={span:5d}{RESET}  {bar}"
                    )
                output = "\n".join(lines) + "\n"
                sys.stdout.write(output)
                sys.stdout.flush()
                displayed_lines = len(lines)

        try:
            await asyncio.gather(watch_stdin(), receive_positions(), display())
        except KeyboardInterrupt:
            stop_event.set()

        # Phase 4: stop recording, save, disconnect
        print()
        print(f"  {CYAN}saving calibration...{RESET}")
        await ws.send(json.dumps({"action": "stop_recording"}))

        # Drain remaining messages until we see 'saved' or hit timeout
        save_path = None
        for _ in range(20):
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = json.loads(raw)
                if msg.get("type") == "saved":
                    save_path = msg.get("path")
                    break
                if msg.get("type") == "error":
                    print(f"{RED}save error: {msg.get('message')}{RESET}", file=sys.stderr)
                    return 1
            except asyncio.TimeoutError:
                break

        try:
            await ws.send(json.dumps({"action": "disconnect"}))
        except Exception:
            pass

        if save_path:
            print(f"  {GREEN}saved at:{RESET} {save_path}")
            return 0
        print(f"{YELLOW}calibration completed but no save path returned.{RESET}", file=sys.stderr)
        return 0


def cmd_calibrate_manual(args):
    # Auto-detect device_type from device_id if not explicit
    device_type = args.device_type
    if device_type == "auto":
        device_type = "teleoperator" if "leader" in args.device_id.lower() else "robot"
    robot_type = args.type
    if robot_type == "auto":
        robot_type = "so101_leader" if device_type == "teleoperator" else "so101_follower"

    print(f"{CYAN}manual cal{RESET}  port={args.port}  device_id={args.device_id}  "
          f"type={robot_type}  device_type={device_type}")
    try:
        rc = asyncio.run(_manual_cal_loop(args.port, args.device_id, device_type, robot_type))
    except KeyboardInterrupt:
        print(f"\n{YELLOW}interrupted.{RESET}")
        rc = 130
    sys.exit(rc)


# ---------------------------------------------------------------------------
# Subcommands — config
# ---------------------------------------------------------------------------

def cmd_config_show(args):
    _pretty(_get("/api/config/"))


def cmd_config_set(args):
    body = _kv_to_body(args.pairs)
    _pretty(_post("/api/config/", body=body))


# ---------------------------------------------------------------------------
# Subcommands — record / teleop / train (lifecycle: start, status, stop)
# ---------------------------------------------------------------------------

def _make_process_lifecycle(name: str, start_path: str, status_path: str, stop_path: str):
    def cmd_start(args):
        body = _kv_to_body(args.pairs)
        result = _post(start_path, body=body, timeout=120)
        process_id = result.get("process_id") or result.get("job_id") or ""
        _pretty(result)
        if process_id and not args.no_stream:
            try:
                asyncio.run(_stream_logs(process_id))
            except KeyboardInterrupt:
                print(f"\n{YELLOW}interrupted — stopping {name} {process_id}...{RESET}")
                try:
                    _post(stop_path.format(id=process_id))
                except Exception:
                    pass

    def cmd_status(args):
        _pretty(_get(status_path.format(id=args.id)))

    def cmd_stop(args):
        _pretty(_post(stop_path.format(id=args.id)))

    return cmd_start, cmd_status, cmd_stop


cmd_record_start, cmd_record_status, cmd_record_stop = _make_process_lifecycle(
    "record", "/api/recording/start", "/api/recording/status/{id}", "/api/recording/stop/{id}"
)
cmd_teleop_start, cmd_teleop_status, cmd_teleop_stop = _make_process_lifecycle(
    "teleop", "/api/teleoperation/start", "/api/teleoperation/status/{id}", "/api/teleoperation/stop/{id}"
)
cmd_train_start, cmd_train_status, cmd_train_cancel = _make_process_lifecycle(
    "train", "/api/training/start", "/api/training/status/{id}", "/api/training/cancel/{id}"
)


def cmd_record_cache_clear(args):
    _pretty(_delete("/api/recording/cache"))


# ---------------------------------------------------------------------------
# Subcommands — HF + system
# ---------------------------------------------------------------------------

def cmd_hf_whoami(args):
    _pretty(_get("/api/huggingface/whoami"))


def cmd_hf_repos(args):
    _pretty(_get("/api/huggingface/repos"))


def cmd_processes(args):
    _pretty(_get("/api/system/status"))


def cmd_locks(args):
    _pretty(_get("/api/system/port-locks"))


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mm", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health").set_defaults(func=cmd_health)
    sub.add_parser("ports").set_defaults(func=cmd_ports)

    pc = sub.add_parser("cameras")
    pc.add_argument("--preview", nargs="*", metavar="INDEX", help="capture preview JPEGs (specify indices or none for all)")
    pc.set_defaults(func=cmd_cameras)

    pw = sub.add_parser("wiggle")
    pw.add_argument("port")
    pw.set_defaults(func=cmd_wiggle)

    cal = sub.add_parser("calibrate").add_subparsers(dest="cal_cmd", required=True)
    cal.add_parser("status").set_defaults(func=cmd_calibrate_status)
    cal.add_parser("missing").set_defaults(func=cmd_calibrate_missing)

    cal_auto = cal.add_parser("auto", help="run full auto-calibration with live log stream")
    cal_auto.add_argument("port")
    cal_auto.add_argument("device_id")
    cal_auto.add_argument("--type", default="so101_follower",
                          choices=["so101_follower", "so100_follower"],
                          help="robot type for the saved cal file (default so101_follower)")
    cal_auto.add_argument("--no-complete", action="store_true",
                          help="skip the post-cal copy step")
    cal_auto.set_defaults(func=cmd_calibrate_auto)

    cal_stop = cal.add_parser("stop")
    cal_stop.add_argument("process_id")
    cal_stop.set_defaults(func=cmd_calibrate_stop)

    cal_man = cal.add_parser("manual", help="interactive 4-phase manual calibration via WebSocket")
    cal_man.add_argument("port")
    cal_man.add_argument("device_id")
    cal_man.add_argument("--type", default="auto",
                         choices=["auto", "so101_leader", "so101_follower"],
                         help="auto = infer from device_id (default)")
    cal_man.add_argument("--device-type", default="auto",
                         choices=["auto", "teleoperator", "robot"],
                         help="auto = infer from device_id (default)")
    cal_man.set_defaults(func=cmd_calibrate_manual)

    cfg = sub.add_parser("config").add_subparsers(dest="cfg_cmd", required=True)
    cfg.add_parser("show").set_defaults(func=cmd_config_show)
    cfg_set = cfg.add_parser("set")
    cfg_set.add_argument("pairs", nargs="+", metavar="KEY=VAL")
    cfg_set.set_defaults(func=cmd_config_set)

    record_subparsers = None
    for name, (start_fn, status_fn, stop_fn), stop_label in [
        ("record", (cmd_record_start, cmd_record_status, cmd_record_stop), "stop"),
        ("teleop", (cmd_teleop_start, cmd_teleop_status, cmd_teleop_stop), "stop"),
        ("train",  (cmd_train_start,  cmd_train_status,  cmd_train_cancel), "cancel"),
    ]:
        grp = sub.add_parser(name).add_subparsers(dest=f"{name}_cmd", required=True)
        s = grp.add_parser("start")
        s.add_argument("pairs", nargs="*", metavar="KEY=VAL", help="request body fields")
        s.add_argument("--no-stream", action="store_true", help="don't stream logs after start")
        s.set_defaults(func=start_fn)
        st = grp.add_parser("status")
        st.add_argument("id", metavar="PROCESS_ID_OR_JOB_ID")
        st.set_defaults(func=status_fn)
        sp = grp.add_parser(stop_label)
        sp.add_argument("id", metavar="PROCESS_ID_OR_JOB_ID")
        sp.set_defaults(func=stop_fn)
        if name == "record":
            record_subparsers = grp

    # `mm record cache clear`
    if record_subparsers is not None:
        cache = record_subparsers.add_parser("cache").add_subparsers(dest="cache_cmd", required=True)
        cache.add_parser("clear").set_defaults(func=cmd_record_cache_clear)

    hf = sub.add_parser("hf").add_subparsers(dest="hf_cmd", required=True)
    hf.add_parser("whoami").set_defaults(func=cmd_hf_whoami)
    hf.add_parser("repos").set_defaults(func=cmd_hf_repos)

    sub.add_parser("processes").set_defaults(func=cmd_processes)
    sub.add_parser("locks").set_defaults(func=cmd_locks)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
        return 0
    except requests.ConnectionError:
        print(f"{RED}Cannot reach backend at {MM_HOST}. Is it running?{RESET}", file=sys.stderr)
        print(f"  start it with: cd MakerMods-App && python -m backend.main", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"{RED}HTTP error: {e}{RESET}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
