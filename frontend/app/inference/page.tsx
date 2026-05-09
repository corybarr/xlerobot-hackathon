"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Brain, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import { LogViewer } from "@/components/common/log-viewer";
import { Alert, AlertDescription } from "@/components/ui/alert";

const API_BASE = "/api/inference/molmo";

function wsUrl(path: string): string {
  if (typeof window === "undefined") return "";
  const base =
    process.env.NEXT_PUBLIC_WS_URL ||
    `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`;
  return `${base}${path}`;
}

type LoadStatus = "unloaded" | "loading" | "ready" | "error";

interface MolmoStatus {
  loaded: boolean;
  ready: boolean;
  device: string;
  warmup_progress: number;
  load_error?: string | null;
}

interface LoopStatus {
  running: boolean;
  step_count: number;
  last_action: number[] | null;
  last_latency_ms: number;
  task: string;
  action_dim: number;
}

function latencyClass(ms: number): string {
  if (ms < 600) return "text-green-600 dark:text-green-400";
  if (ms <= 1000) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

export default function MolmoInferencePage() {
  const [device, setDevice] = useState<"cuda" | "cpu">("cuda");
  const [remoteHost, setRemoteHost] = useState("");
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>("unloaded");
  const [statusDetail, setStatusDetail] = useState<MolmoStatus | null>(null);

  const [task, setTask] = useState(
    "Set the table: place plate, fork on left, knife on right"
  );
  const [hz, setHz] = useState(2);
  const [camTop, setCamTop] = useState(0);
  const [camSide, setCamSide] = useState(1);
  const [robotPort, setRobotPort] = useState("/dev/ttyACM0");
  const [dryRun, setDryRun] = useState(false);
  const [loopBusy, setLoopBusy] = useState(false);
  const [loopError, setLoopError] = useState<string | null>(null);

  const [loopStatus, setLoopStatus] = useState<LoopStatus | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/status`);
      const j = (await r.json()) as MolmoStatus;
      setStatusDetail(j);
      if (j.load_error) {
        setLoadStatus("error");
      } else if (j.ready) {
        setLoadStatus("ready");
      } else if (j.loaded || j.warmup_progress > 0) {
        setLoadStatus("loading");
      } else {
        setLoadStatus("unloaded");
      }
    } catch {
      /* ignore */
    }
  }, []);

  const fetchLoopStatus = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/loop_status`);
      if (r.status === 400) return;
      const j = (await r.json()) as LoopStatus;
      setLoopStatus(j);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    const id = window.setInterval(fetchStatus, 2000);
    return () => window.clearInterval(id);
  }, [fetchStatus]);

  useEffect(() => {
    if (!loopStatus?.running) return;
    const id = window.setInterval(fetchLoopStatus, 1000);
    return () => window.clearInterval(id);
  }, [loopStatus?.running, fetchLoopStatus]);

  useEffect(() => {
    const url = wsUrl("/ws/inference/molmo/logs");
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onerror = () => setWsConnected(false);
    ws.onmessage = (ev) => {
      setLogs((prev) => [...prev.slice(-499), ev.data as string]);
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, []);

  async function handleLoad() {
    setLoadBusy(true);
    setLoopError(null);
    try {
      if (remoteHost.trim()) {
        const r = await fetch(`${API_BASE}/load`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device, remote_host: remoteHost.trim() }),
        });
        const errBody = await r.json().catch(() => ({}));
        const det = (errBody as { detail?: unknown }).detail;
        const remoteMsg =
          typeof det === "object" &&
          det !== null &&
          "error" in det &&
          typeof (det as { error: unknown }).error === "string"
            ? (det as { error: string }).error
            : "Remote inference is not available yet.";
        if (r.status === 501) {
          setLoopError(remoteMsg);
          setLoadStatus("error");
          return;
        }
        setLoopError(remoteMsg);
        setLoadStatus("error");
        return;
      }
      const r = await fetch(`${API_BASE}/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        const d = (j as { detail?: unknown }).detail;
        const msg =
          typeof d === "string"
            ? d
            : d && typeof d === "object" && "error" in d
              ? String((d as { error: string }).error)
              : r.statusText;
        throw new Error(msg);
      }
      const body = await r.json();
      if ((body as { status?: string }).status === "already_loaded") {
        setLoadStatus("ready");
        await fetchStatus();
        return;
      }
      setLoadStatus("loading");
      await fetchStatus();
    } catch (e) {
      setLoopError(e instanceof Error ? e.message : "Load failed");
      setLoadStatus("error");
    } finally {
      setLoadBusy(false);
    }
  }

  async function handleStart() {
    setLoopBusy(true);
    setLoopError(null);
    try {
      const r = await fetch(`${API_BASE}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task: task.trim(),
          camera_indices: [camTop, camSide],
          robot_port: dryRun ? "" : robotPort,
          hz,
          dry_run: dryRun,
        }),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        const d = (j as { detail?: unknown }).detail;
        let msg = r.statusText;
        if (typeof d === "string") msg = d;
        else if (d && typeof d === "object" && "error" in d)
          msg = String((d as { error: string }).error);
        throw new Error(msg);
      }
      await fetchLoopStatus();
    } catch (e) {
      setLoopError(e instanceof Error ? e.message : "Start failed");
    } finally {
      setLoopBusy(false);
    }
  }

  async function handleStop() {
    setLoopBusy(true);
    try {
      await fetch(`${API_BASE}/stop`, { method: "POST" });
      setLoopStatus((s) =>
        s
          ? {
              ...s,
              running: false,
            }
          : null
      );
    } finally {
      setLoopBusy(false);
    }
  }

  const ready = loadStatus === "ready" && statusDetail?.ready;
  const running = loopStatus?.running ?? false;
  const warmupLabel =
    statusDetail && !statusDetail.ready && statusDetail.loaded
      ? `Warming up (${statusDetail.warmup_progress}/5)`
      : null;

  return (
    <div className="min-h-screen bg-muted px-6 py-10">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/" className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              Wizard
            </Link>
          </Button>
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary" />
            <h1 className="text-xl font-semibold tracking-tight">
              MolmoAct2 inference
            </h1>
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Model load</CardTitle>
            <CardDescription>
              Load allenai/MolmoAct2-SO100_101 (GPU recommended). CUDA graph warmup
              runs automatically.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Device</Label>
              <RadioGroup
                value={device}
                onValueChange={(v) => setDevice(v as "cuda" | "cpu")}
                className="flex gap-4"
                disabled={loadBusy}
              >
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="cuda" id="dev-cuda" />
                  <Label htmlFor="dev-cuda">CUDA</Label>
                </div>
                <div className="flex items-center gap-2">
                  <RadioGroupItem value="cpu" id="dev-cpu" />
                  <Label htmlFor="dev-cpu">CPU</Label>
                </div>
              </RadioGroup>
            </div>
            <div className="space-y-2">
              <Label htmlFor="remote">Remote host (optional, IP:port)</Label>
              <Input
                id="remote"
                placeholder="Leave blank for local load"
                value={remoteHost}
                onChange={(e) => setRemoteHost(e.target.value)}
                disabled={loadBusy}
              />
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button onClick={handleLoad} disabled={loadBusy}>
                {loadBusy ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : null}
                Load model
              </Button>
              <span className="text-sm text-muted-foreground">
                Status:{" "}
                {loadStatus === "unloaded" && "Unloaded"}
                {loadStatus === "loading" &&
                  (warmupLabel || "Loading…")}
                {loadStatus === "ready" && "Ready"}
                {loadStatus === "error" && "Error"}
              </span>
            </div>
            {statusDetail?.load_error && (
              <Alert variant="destructive">
                <AlertDescription>{statusDetail.load_error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card className={!ready ? "opacity-60" : ""}>
          <CardHeader>
            <CardTitle>Task & control</CardTitle>
            <CardDescription>
              Runs a live loop: cameras → model → joint targets.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!ready && (
              <Alert>
                <AlertDescription>
                  Load the model and wait until status is Ready.
                </AlertDescription>
              </Alert>
            )}
            <div className="space-y-2">
              <Label htmlFor="task">Task instruction</Label>
              <Input
                id="task"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                disabled={!ready || running}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="hz">Control Hz</Label>
                <Input
                  id="hz"
                  type="number"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={hz}
                  onChange={(e) => setHz(parseFloat(e.target.value) || 2)}
                  disabled={!ready || running}
                />
              </div>
              <div className="flex items-end gap-3 pb-2">
                <Switch
                  id="dry"
                  checked={dryRun}
                  onCheckedChange={setDryRun}
                  disabled={!ready || running}
                />
                <Label htmlFor="dry">Dry run (no robot)</Label>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="cam0">Top camera index</Label>
                <Input
                  id="cam0"
                  type="number"
                  value={camTop}
                  onChange={(e) => setCamTop(parseInt(e.target.value, 10) || 0)}
                  disabled={!ready || running}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="cam1">Side camera index</Label>
                <Input
                  id="cam1"
                  type="number"
                  value={camSide}
                  onChange={(e) => setCamSide(parseInt(e.target.value, 10) || 0)}
                  disabled={!ready || running}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="port">Robot port</Label>
              <Input
                id="port"
                value={robotPort}
                onChange={(e) => setRobotPort(e.target.value)}
                disabled={!ready || running || dryRun}
              />
            </div>
            {loopError && (
              <Alert variant="destructive">
                <AlertDescription>{loopError}</AlertDescription>
              </Alert>
            )}
            <div className="flex gap-2">
              {!running ? (
                <Button
                  onClick={handleStart}
                  disabled={!ready || loopBusy || !task.trim()}
                >
                  {loopBusy ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : null}
                  Start inference
                </Button>
              ) : (
                <Button
                  variant="outline"
                  onClick={handleStop}
                  disabled={loopBusy}
                >
                  Stop inference
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {running && loopStatus && (
          <Card>
            <CardHeader>
              <CardTitle>Live status</CardTitle>
              <CardDescription>{loopStatus.task}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap gap-6 text-sm">
                <div>
                  <span className="text-muted-foreground">Step</span>{" "}
                  <span className="font-mono font-medium">
                    {loopStatus.step_count}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Latency</span>{" "}
                  <span
                    className={`font-mono font-medium ${latencyClass(loopStatus.last_latency_ms)}`}
                  >
                    {loopStatus.last_latency_ms.toFixed(0)} ms
                  </span>
                </div>
              </div>
              {loopStatus.last_action && loopStatus.last_action.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">
                    Last action (normalized)
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {loopStatus.last_action.map((v, i) => (
                      <div
                        key={i}
                        className="flex flex-col items-center gap-1 rounded border bg-background px-2 py-1"
                      >
                        <div
                          className="w-8 rounded-sm bg-primary/30"
                          style={{
                            height: `${Math.min(100, Math.abs(v))}px`,
                          }}
                        />
                        <span className="font-mono text-[10px] text-muted-foreground">
                          j{i}: {v.toFixed(1)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Inference log</CardTitle>
            <CardDescription>Streamed from the control loop</CardDescription>
          </CardHeader>
          <CardContent>
            <LogViewer
              logs={logs}
              isConnected={wsConnected}
              onClear={() => setLogs([])}
              maxHeight="320px"
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
