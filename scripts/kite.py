#!/usr/bin/env python3
"""kite — CLI wrapper for the KiteML REST API (api.kiteml.com).

Drives KiteML's full workflow from terminal: projects, datasets (HF import),
policies, training jobs, artifact download. Reads the bearer token from
$KITE_TOKEN env or ~/.kite_token (mode 600); auth is `Authorization: Bearer`.

Subcommands:
  whoami                                — auth + reachability check
  projects list                         — list your projects
  projects create <name>                — create a project
  projects delete <id>
  robots                                — registered robot embodiments
  policies                              — available policy registry (smolvla, pi0, ...)
  datasets list                         — your imported datasets
  datasets import <hf_repo> [--project <id>]
                                        — import HF dataset (Mattie-NT/...)
  datasets get <id>
  datasets delete <id>
  train configure <project_id> <policy_id> [--method <m>] [--hardware <tier>]
                                        — auto-configure training params
  train start <project_id> <config_json_or_file>
                                        — create training job from config
  train list                            — list jobs
  train status <job_id>                 — job status
  train logs <job_id>                   — stream job logs
  train stop <job_id>
  train checkpoints <job_id>            — list saved checkpoints
  train download <job_id> <type>        — download an artifact
  api-keys list / create <name> / revoke <id>

Env:
  KITE_TOKEN     Bearer token; falls back to ~/.kite_token
  KITE_HOST      default https://api.kiteml.com

Examples:
  ./scripts/kite.py whoami
  ./scripts/kite.py projects create "set-the-table"
  ./scripts/kite.py datasets import Mattie-NT/makermods_pick_cup_combined --project <pid>
  ./scripts/kite.py train configure <pid> smolvla
  ./scripts/kite.py train start <pid> @/tmp/cfg.json
  ./scripts/kite.py train logs <job_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

KITE_HOST = os.getenv("KITE_HOST", "https://api.kiteml.com").rstrip("/")
TOKEN_FILE = Path.home() / ".kite_token"

RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"; DIM = "\033[2m"; RESET = "\033[0m"


def _load_token() -> str:
    tok = os.getenv("KITE_TOKEN", "").strip()
    if tok:
        return tok
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    print(f"{RED}no token. set KITE_TOKEN env or write to ~/.kite_token{RESET}", file=sys.stderr)
    sys.exit(1)


def _headers() -> dict:
    return {"Authorization": f"Bearer {_load_token()}"}


def _url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    # Auto-prefix /api unless the path already targets /api/* (note the slash —
    # /api-keys must NOT match /api here).
    if not (path.startswith("/api/") or path == "/api"):
        path = "/api" + path
    return f"{KITE_HOST}{path}"


def _get(path: str, params: Optional[dict] = None, timeout: int = 30) -> Any:
    r = requests.get(_url(path), headers=_headers(), params=params, timeout=timeout)
    if not r.ok:
        _err(r)
    return r.json() if r.content else None


def _post(path: str, body: Any = None, timeout: int = 60) -> Any:
    r = requests.post(_url(path), headers=_headers(), json=body, timeout=timeout)
    if not r.ok:
        _err(r)
    return r.json() if r.content else None


def _delete(path: str, timeout: int = 30) -> Any:
    r = requests.delete(_url(path), headers=_headers(), timeout=timeout)
    if not r.ok:
        _err(r)
    return r.json() if r.content else None


def _err(r: requests.Response):
    try:
        body = r.json()
        print(f"{RED}HTTP {r.status_code}: {json.dumps(body)}{RESET}", file=sys.stderr)
    except Exception:
        print(f"{RED}HTTP {r.status_code}: {r.text[:500]}{RESET}", file=sys.stderr)
    sys.exit(2)


def _pretty(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _items(resp: Any, *keys: str) -> list:
    """Some endpoints return list, some return {items:[...]}, etc. Normalize."""
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in keys:
            v = resp.get(k)
            if isinstance(v, list):
                return v
        # last resort
        for v in resp.values():
            if isinstance(v, list):
                return v
    return []


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_whoami(args):
    # No /me endpoint exists; use /api/datasets as the auth probe.
    ds = _get("/datasets")
    pj = _get("/projects")
    print(f"{GREEN}auth ok{RESET}  host={KITE_HOST}")
    print(f"  datasets:  {len(_items(ds, 'datasets', 'items'))}")
    print(f"  projects:  {len(_items(pj, 'projects', 'items'))}")


def cmd_projects_list(args):
    items = _items(_get("/projects"), "projects", "items")
    if not items:
        print(f"{YELLOW}(no projects){RESET}")
        return
    for p in items:
        print(f"  {CYAN}{p.get('id','?')}{RESET}  name={p.get('name','?')}  created={(p.get('created_at') or '')[:10]}")


def cmd_projects_create(args):
    body = {"name": args.name}
    if args.config_json:
        body.update(json.loads(args.config_json))
    result = _post("/projects", body=body)
    print(f"{GREEN}created{RESET}")
    _pretty(result)


def cmd_projects_delete(args):
    _pretty(_delete(f"/projects/{args.project_id}"))


def cmd_robots(args):
    items = _items(_get("/models/robots"), "robots", "items")
    for r in items:
        print(f"  {CYAN}{r.get('id','?'):28s}{RESET}  {r.get('name','?'):28s}  type={r.get('type','?')}")


def cmd_policies(args):
    items = _items(_get("/policies/registry"), "policies", "items")
    for p in items:
        print(f"  {CYAN}{p.get('id','?'):20s}{RESET}  name={p.get('name','?')}  family={p.get('family','?')}")


def cmd_datasets_list(args):
    items = _items(_get("/datasets"), "datasets", "items")
    if not items:
        print(f"{YELLOW}(no datasets imported yet){RESET}")
        return
    for d in items:
        print(f"  {CYAN}{d.get('id','?')}{RESET}  source={d.get('source','?')}  episodes={d.get('total_episodes','?')}  status={d.get('status','?')}")


def cmd_datasets_import(args):
    body: dict = {"input": args.source}
    if args.project:
        body["project_id"] = args.project
    if args.credentials:
        body["credentials_id"] = args.credentials
    print(f"  importing {args.source} ...")
    result = _post("/datasets/import", body=body, timeout=180)
    _pretty(result)


def cmd_datasets_get(args):
    _pretty(_get(f"/datasets/{args.dataset_id}"))


def cmd_datasets_delete(args):
    _pretty(_delete(f"/datasets/{args.dataset_id}"))


def cmd_train_configure(args):
    body: dict = {"project_id": args.project_id, "policy_id": args.policy_id}
    if args.method:
        body["training_method"] = args.method
    if args.hardware:
        body["hardware_tier"] = args.hardware
    if args.task:
        body["task_description"] = args.task
    if args.overrides_json:
        body["overrides"] = json.loads(args.overrides_json)
    _pretty(_post("/training/configure", body=body))


def cmd_train_start(args):
    if args.config.startswith("@"):
        config = json.loads(Path(args.config[1:]).read_text())
    else:
        config = json.loads(args.config)
    body: dict = {"project_id": args.project_id, "config": config}
    if args.scene_json:
        body["scene"] = json.loads(args.scene_json)
    print(f"  starting training job in project {args.project_id} ...")
    result = _post("/training/jobs", body=body)
    _pretty(result)


def cmd_train_list(args):
    items = _items(_get("/training/jobs"), "jobs", "items")
    if not items:
        print(f"{YELLOW}(no training jobs){RESET}")
        return
    for j in items:
        st = j.get("status", "?")
        st_color = GREEN if st == "completed" else (YELLOW if st in ("running","queued") else (RED if st == "failed" else ""))
        print(f"  {CYAN}{j.get('id','?')}{RESET}  policy={j.get('policy_id','?'):12s}  {st_color}{st}{RESET}  created={(j.get('created_at') or '')[:16]}")


def cmd_train_status(args):
    _pretty(_get(f"/training/jobs/{args.job_id}"))


def cmd_train_stop(args):
    _pretty(_post(f"/training/jobs/{args.job_id}/stop"))


def cmd_train_logs(args):
    """Poll the logs endpoint and print new lines as they appear."""
    last = 0
    try:
        while True:
            data = _get(f"/training/jobs/{args.job_id}/logs",
                        params={"offset": last} if args.follow else None)
            lines = _items(data, "logs", "lines")
            if isinstance(data, dict) and "text" in data:
                # Some servers return {text: "..."} blob
                text = data.get("text", "")
                new_text = text[last:]
                if new_text:
                    print(new_text, end="", flush=True)
                    last = len(text)
            else:
                for line in lines:
                    if isinstance(line, dict):
                        print(line.get("message") or line.get("text") or json.dumps(line), flush=True)
                    else:
                        print(line, flush=True)
                last += len(lines)
            # Check job status — exit when done
            status = _get(f"/training/jobs/{args.job_id}").get("status", "?")
            if status in ("completed", "failed", "cancelled"):
                print(f"\n{DIM}--- job {status} ---{RESET}")
                return
            if not args.follow:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}stopped following (job continues){RESET}")


def cmd_train_checkpoints(args):
    _pretty(_get(f"/training/jobs/{args.job_id}/checkpoints"))


def cmd_train_artifacts(args):
    _pretty(_get(f"/training/jobs/{args.job_id}/artifacts"))


def cmd_train_download(args):
    url = _url(f"/training/jobs/{args.job_id}/artifacts/{args.artifact_type}/download")
    print(f"  downloading {url} -> {args.output}")
    with requests.get(url, headers=_headers(), stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(args.output, "wb") as f:
            n = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                n += len(chunk)
                if total:
                    pct = 100 * n / total
                    print(f"\r  {n / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print(f"\n{GREEN}done{RESET}")


def cmd_keys_list(args):
    _pretty(_get("/api-keys"))


def cmd_keys_create(args):
    _pretty(_post("/api-keys", body={"name": args.name}))


def cmd_keys_revoke(args):
    _pretty(_delete(f"/api-keys/{args.key_id}"))


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kite", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami").set_defaults(func=cmd_whoami)
    sub.add_parser("robots").set_defaults(func=cmd_robots)
    sub.add_parser("policies").set_defaults(func=cmd_policies)

    pj = sub.add_parser("projects").add_subparsers(dest="pj_cmd", required=True)
    pj.add_parser("list").set_defaults(func=cmd_projects_list)
    pj_create = pj.add_parser("create"); pj_create.add_argument("name")
    pj_create.add_argument("--config-json", help="JSON for extra fields"); pj_create.set_defaults(func=cmd_projects_create)
    pj_del = pj.add_parser("delete"); pj_del.add_argument("project_id"); pj_del.set_defaults(func=cmd_projects_delete)

    ds = sub.add_parser("datasets").add_subparsers(dest="ds_cmd", required=True)
    ds.add_parser("list").set_defaults(func=cmd_datasets_list)
    ds_imp = ds.add_parser("import")
    ds_imp.add_argument("source", help="HuggingFace repo ID like Mattie-NT/makermods_pick_cup_combined or s3:// URI")
    ds_imp.add_argument("--project", help="link to this project")
    ds_imp.add_argument("--credentials", help="credential ID for s3 sources")
    ds_imp.set_defaults(func=cmd_datasets_import)
    ds_get = ds.add_parser("get"); ds_get.add_argument("dataset_id"); ds_get.set_defaults(func=cmd_datasets_get)
    ds_del = ds.add_parser("delete"); ds_del.add_argument("dataset_id"); ds_del.set_defaults(func=cmd_datasets_delete)

    tr = sub.add_parser("train").add_subparsers(dest="tr_cmd", required=True)
    tr_cfg = tr.add_parser("configure")
    tr_cfg.add_argument("project_id"); tr_cfg.add_argument("policy_id")
    tr_cfg.add_argument("--method"); tr_cfg.add_argument("--hardware")
    tr_cfg.add_argument("--task", help="task description"); tr_cfg.add_argument("--overrides-json")
    tr_cfg.set_defaults(func=cmd_train_configure)
    tr_start = tr.add_parser("start")
    tr_start.add_argument("project_id"); tr_start.add_argument("config", help="JSON string OR @path/to/cfg.json")
    tr_start.add_argument("--scene-json")
    tr_start.set_defaults(func=cmd_train_start)
    tr.add_parser("list").set_defaults(func=cmd_train_list)
    tr_status = tr.add_parser("status"); tr_status.add_argument("job_id"); tr_status.set_defaults(func=cmd_train_status)
    tr_stop = tr.add_parser("stop"); tr_stop.add_argument("job_id"); tr_stop.set_defaults(func=cmd_train_stop)
    tr_logs = tr.add_parser("logs")
    tr_logs.add_argument("job_id"); tr_logs.add_argument("--follow", "-f", action="store_true")
    tr_logs.add_argument("--interval", type=float, default=3.0)
    tr_logs.set_defaults(func=cmd_train_logs)
    tr_ckpt = tr.add_parser("checkpoints"); tr_ckpt.add_argument("job_id"); tr_ckpt.set_defaults(func=cmd_train_checkpoints)
    tr_art = tr.add_parser("artifacts"); tr_art.add_argument("job_id"); tr_art.set_defaults(func=cmd_train_artifacts)
    tr_dl = tr.add_parser("download")
    tr_dl.add_argument("job_id"); tr_dl.add_argument("artifact_type"); tr_dl.add_argument("output")
    tr_dl.set_defaults(func=cmd_train_download)

    keys = sub.add_parser("api-keys").add_subparsers(dest="keys_cmd", required=True)
    keys.add_parser("list").set_defaults(func=cmd_keys_list)
    keys_create = keys.add_parser("create"); keys_create.add_argument("name"); keys_create.set_defaults(func=cmd_keys_create)
    keys_revoke = keys.add_parser("revoke"); keys_revoke.add_argument("key_id"); keys_revoke.set_defaults(func=cmd_keys_revoke)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
        return 0
    except requests.ConnectionError:
        print(f"{RED}cannot reach KiteML at {KITE_HOST}{RESET}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
