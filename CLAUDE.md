# MakerMods LeRobot UI

Web UI for xLeRobot SO101 bimanual robot arms — teleoperation, calibration, and data recording.

## Architecture

- **Backend**: FastAPI (Python) at `backend/` — wraps lerobot CLI commands via subprocess
- **Frontend**: Next.js 16 (App Router) at `frontend/` — wizard-style setup flow
- **State**: React Context (client) + `webui_config.json` (backend persistence)
- **Communication**: REST API + WebSocket for real-time log streaming
- Frontend proxies `/api/*` and `/ws/*` to the backend via Next.js rewrites in `frontend/next.config.ts`

## Prerequisites

- Python with lerobot installed (`conda activate lerobot`)
- Node.js for the frontend

## Running

```bash
# Terminal 1: Backend (port 8000)
conda activate lerobot
cd /path/to/MakerMods-LeRobot-UI
python -m backend.main

# Terminal 2: Frontend (port 3000)
cd frontend
npm install   # first time only
npm run dev

# Open: http://localhost:3000
```

## Backend

### Import conventions
- Internal imports use `from backend.*` (e.g. `from backend.services.config_manager import ConfigManager`)
- `lerobot.motors.*` imports reference the actual lerobot library (must be installed)

### Key paths
- `backend/main.py` — FastAPI app, CORS, router registration, static file mounts
- `backend/api/` — Route handlers (setup, calibration, teleoperation, recording, config, huggingface, system)
- `backend/models/` — Pydantic models (config, setup, system, recording, teleoperation)
- `backend/services/` — Business logic (config_manager, process_manager, port_scanner, camera_scanner, calibration_service, hf_service, manual_calibration, auto_calibration)
- `backend/websockets/logs.py` — WebSocket log streaming
- `webui_config.json` — Persisted config at repo root (gitignored)

### Path resolution
- `repo_root` in `main.py` = `Path(__file__).parent.parent` (repo root)
- `repo_root` in `config_manager.py` = `Path(__file__).parent.parent.parent` (repo root)

## Frontend

### Stack
- Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (new-york style)
- Path aliases: `@/components`, `@/lib`, `@/hooks`

### Key paths
- `frontend/app/page.tsx` — Single-page wizard host
- `frontend/components/wizard/` — Wizard layout, provider (React Context + reducer), sidebar, topbar, step-card
- `frontend/components/wizard/steps/` — 6 steps: robot-type, ports, cameras, calibration, teleoperate, record
- `frontend/components/common/` — Shared components (log-viewer, process-status, dev-error-panel, robot-display)
- `frontend/components/ui/` — shadcn/ui components (do not edit manually, use `npx shadcn add`)
- `frontend/lib/api.ts` — Real API client
- `frontend/lib/services.ts` — Service layer (has `USE_MOCK` toggle)
- `frontend/lib/wizard-types.ts` — TypeScript interfaces, constants, initial state
- `frontend/hooks/` — Custom hooks (use-websocket, use-manual-calibration)

## Development notes

- When implementing a new feature or fixing a bug, update `PROGRESS.md` with a dated changelog entry
- Backend wraps lerobot CLI commands via subprocess — zero changes to lerobot core code
- Robot types: `so101_follower`, `bi_so101_follower`, `so101_leader`, `bi_so101_leader`
- Calibration files live at `~/.cache/huggingface/lerobot/calibration/{robots|teleoperators}/{type}/{id}.json`
- Don't cast Pydantic-derived interfaces with `as Record<string, unknown>` — use spread operator instead
