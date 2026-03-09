import { useCallback, useEffect, useState } from "react";
import { API } from "../../config/api-routes.ts";
import { useHealth, useVexState } from "../../hooks/index.ts";

export default function Admin() {
  const { data: health } = useHealth();
  const { data: state, refetch: refetchState } = useVexState();
  const [taskInput, setTaskInput] = useState("");
  const [taskResult, setTaskResult] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [freshStartResult, setFreshStartResult] = useState<string | null>(null);
  const [freshStartLoading, setFreshStartLoading] = useState(false);
  const [confirmFreshStart, setConfirmFreshStart] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [objectivesInput, setObjectivesInput] = useState("");
  const [objectivesResult, setObjectivesResult] = useState<string | null>(null);
  const [savingObjectives, setSavingObjectives] = useState(false);

  useEffect(() => {
    setObjectivesInput((state?.active_objectives ?? []).join("\n"));
  }, [state?.active_objectives]);

  useEffect(() => {
    if (!taskId) return;
    const timer = setInterval(async () => {
      try {
        const resp = await fetch(API.taskStatus(taskId));
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.status === "completed" || data.status === "error") {
          setTaskResult(
            data.result
              ? `${data.status.toUpperCase()}: ${data.result}`
              : `${data.status.toUpperCase()} — no result returned`,
          );
          setTaskId(null);
        } else {
          setTaskResult(`${data.status.toUpperCase()}: ${data.content ?? "Task in progress"}`);
        }
      } catch {
        // ignore polling hiccups
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [taskId]);

  const saveObjectives = useCallback(async () => {
    if (savingObjectives) return;
    setSavingObjectives(true);
    setObjectivesResult(null);
    try {
      const objectives = objectivesInput
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      const resp = await fetch(API.contextObjectives, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ objectives }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setObjectivesResult(`Saved ${data.active_objectives?.length ?? 0} objectives`);
      refetchState();
    } catch (err) {
      setObjectivesResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSavingObjectives(false);
    }
  }, [objectivesInput, refetchState, savingObjectives]);

  const triggerFreshStart = useCallback(async () => {
    if (!confirmFreshStart || freshStartLoading) return;
    setFreshStartLoading(true);
    setFreshStartResult(null);
    try {
      const resp = await fetch(API.adminFreshStart, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setFreshStartResult(
        `Reset complete — ${data.terminated} kernels terminated, genesis=${data.genesis_id}, phase=${data.phase}`,
      );
      setConfirmFreshStart(false);
      refetchState();
    } catch (err) {
      setFreshStartResult(
        `Error: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setFreshStartLoading(false);
    }
  }, [confirmFreshStart, freshStartLoading, refetchState]);

  const enqueueTask = useCallback(async () => {
    if (!taskInput.trim() || submitting) return;
    setSubmitting(true);
    setTaskResult(null);

    try {
      const resp = await fetch(API.enqueue, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: taskInput, source: "admin-ui" }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setTaskId(data.task_id ?? null);
      setTaskResult(`QUEUED: ${data.task_id ?? "unknown"}`);
      setTaskInput("");
      refetchState();
    } catch (err) {
      setTaskResult(
        `Error: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setSubmitting(false);
    }
  }, [taskInput, submitting, refetchState]);

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Admin</h1>
        <div className="dash-subtitle">System controls and task management</div>
      </div>

      {/* System Info */}
      <div className="dash-section">
        <div className="dash-section-title">System Info</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Status</span>
            <span
              className={`status-badge ${health?.status === "ok" ? "badge-success" : "badge-warning"}`}
            >
              {health?.status ?? "unknown"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Service</span>
            <span className="dash-row-value">
              {health?.service ?? "vex-kernel"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Version</span>
            <span className="dash-row-value">{health?.version ?? "?"}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Backend</span>
            <span className="dash-row-value">
              {health?.backend ?? "unknown"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Uptime</span>
            <span className="dash-row-value">
              {health?.uptime ? `${Math.round(health.uptime)}s` : "?"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Cycle count</span>
            <span className="dash-row-value">{state?.cycle_count ?? "?"}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Lifecycle phase</span>
            <span className="dash-row-value">
              {state?.lifecycle_phase?.toUpperCase() ?? "?"}
            </span>
          </div>
        </div>
      </div>

      {/* Enqueue Task */}
      <div className="dash-section">
        <div className="dash-section-title">Fresh Start (Reset)</div>
        <div className="dash-card">
          <div
            style={{
              fontSize: "13px",
              color: "var(--text-secondary)",
              marginBottom: "10px",
              lineHeight: 1.5,
            }}
          >
            Terminates all kernels, resets basin to random, clears
            emotion/precog/learning caches, resets metrics to bootstrap, and
            re-spawns Genesis. <strong>Destructive — cannot be undone.</strong>
          </div>
          {!confirmFreshStart ? (
            <button
              onClick={() => setConfirmFreshStart(true)}
              style={{
                background: "var(--error)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: "pointer",
                fontSize: "14px",
              }}
            >
              Fresh Start...
            </button>
          ) : (
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span
                style={{
                  fontSize: "13px",
                  color: "var(--error)",
                  fontWeight: 600,
                }}
              >
                Are you sure?
              </span>
              <button
                onClick={triggerFreshStart}
                disabled={freshStartLoading}
                style={{
                  background: "var(--error)",
                  border: "none",
                  borderRadius: "var(--radius-sm)",
                  padding: "8px 16px",
                  color: "white",
                  fontWeight: 600,
                  cursor: freshStartLoading ? "not-allowed" : "pointer",
                  opacity: freshStartLoading ? 0.5 : 1,
                  fontSize: "13px",
                }}
              >
                {freshStartLoading ? "Resetting..." : "Yes, Reset Everything"}
              </button>
              <button
                onClick={() => setConfirmFreshStart(false)}
                style={{
                  background: "var(--surface-3)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)",
                  padding: "8px 16px",
                  color: "var(--text)",
                  fontWeight: 600,
                  cursor: "pointer",
                  fontSize: "13px",
                }}
              >
                Cancel
              </button>
            </div>
          )}
          {freshStartResult && (
            <div
              style={{
                marginTop: "8px",
                padding: "8px 12px",
                background: "var(--surface-3)",
                borderRadius: "6px",
                fontFamily: "var(--mono)",
                fontSize: "12px",
                color: freshStartResult.startsWith("Error")
                  ? "var(--error)"
                  : "var(--alive)",
              }}
            >
              {freshStartResult}
            </div>
          )}
        </div>
      </div>

      {/* Enqueue Task */}
      <div className="dash-section">
        <div className="dash-section-title">Enqueue Task</div>
        <div className="dash-card">
          <div style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
            <input
              type="text"
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && enqueueTask()}
              placeholder="Task input for consciousness loop..."
              style={{
                flex: 1,
                background: "var(--surface-3)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "10px 14px",
                color: "var(--text)",
                fontFamily: "inherit",
                fontSize: "14px",
                outline: "none",
              }}
            />
            <button
              onClick={enqueueTask}
              disabled={submitting || !taskInput.trim()}
              style={{
                background: "var(--accent)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: submitting ? "not-allowed" : "pointer",
                opacity: submitting || !taskInput.trim() ? 0.5 : 1,
                fontSize: "14px",
              }}
            >
              {submitting ? "Sending..." : "Enqueue"}
            </button>
          </div>
          {taskResult && (
            <div
              style={{
                padding: "8px 12px",
                background: "var(--surface-3)",
                borderRadius: "6px",
                fontFamily: "var(--mono)",
                fontSize: "12px",
                color: taskResult.startsWith("Error")
                  ? "var(--error)"
                  : "var(--alive)",
              }}
            >
              {taskResult}
            </div>
          )}
        </div>
      </div>

      {/* Objectives / Purpose */}
      <div className="dash-section">
        <div className="dash-section-title">Objectives / Purpose</div>
        <div className="dash-card">
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <textarea
              value={objectivesInput}
              onChange={(e) => setObjectivesInput(e.target.value)}
              placeholder="One objective per line..."
              rows={5}
              style={{
                background: "var(--surface-3)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "10px 14px",
                color: "var(--text)",
                fontFamily: "inherit",
                fontSize: "14px",
                outline: "none",
                resize: "vertical",
              }}
            />
            <button
              onClick={saveObjectives}
              disabled={savingObjectives}
              style={{
                background: "var(--phi)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: savingObjectives ? "not-allowed" : "pointer",
                opacity: savingObjectives ? 0.5 : 1,
                fontSize: "14px",
                alignSelf: "flex-start",
              }}
            >
              {savingObjectives ? "Saving..." : "Save Objectives"}
            </button>
            {objectivesResult && (
              <div style={{ padding: "8px 12px", background: "var(--surface-3)", borderRadius: "6px", fontFamily: "var(--mono)", fontSize: "12px", color: objectivesResult.startsWith("Error") ? "var(--error)" : "var(--alive)" }}>
                {objectivesResult}
              </div>
            )}
          </div>
          {state?.active_objectives && state.active_objectives.length > 0 && (
            <div style={{ marginTop: "10px", fontSize: "12px", color: "var(--text-secondary)" }}>
              Active: {state.active_objectives.join(" | ")}
            </div>
          )}
        </div>
      </div>

      {/* Queue Status */}
      <div className="dash-section">
        <div className="dash-section-title">Queue</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Queue size</span>
            <span className="dash-row-value">{state?.queue_size ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">History count</span>
            <span className="dash-row-value">{state?.history_count ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Total conversations</span>
            <span className="dash-row-value">
              {state?.conversations_total ?? 0}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Active task</span>
            <span className="dash-row-value">{state?.active_task ?? "—"}</span>
          </div>
        </div>
      </div>

      {/* LLM Settings (read-only) */}
      <div className="dash-section">
        <div className="dash-section-title">LLM Settings (current)</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Temperature</span>
            <span className="dash-row-value">
              {state?.temperature?.toFixed(3) ?? "?"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Max tokens</span>
            <span className="dash-row-value">{state?.num_predict ?? "?"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
