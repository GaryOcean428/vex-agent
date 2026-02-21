import { useCallback, useState } from "react";
import { API } from "../../config/api-routes.ts";
import { usePolledData } from "../../hooks/index.ts";

interface RateLimitEntry {
  current: number;
  limit: number;
  window_seconds: number;
}

interface ForageHistoryEntry {
  timestamp: number;
  query: string;
  results_count: number;
  results: Array<{ title: string; url: string; snippet: string }>;
  summary: string;
}

interface GovernorState {
  enabled: boolean;
  kill_switch: boolean;
  autonomous_search: boolean;
  budget: {
    daily_spend: number;
    daily_ceiling: number;
    budget_remaining: number;
    budget_percent: number;
    call_counts: Record<string, number>;
    last_reset: number;
  };
  rate_limits: Record<string, RateLimitEntry>;
  foraging?: {
    enabled?: boolean;
    forage_count?: number;
    max_daily?: number;
    cooldown_remaining?: number;
    last_query?: string | null;
    last_summary?: string | null;
    history?: ForageHistoryEntry[];
  };
}

function useGovernor() {
  return usePolledData<GovernorState>(API.governor, 3000);
}

export default function Governor() {
  const { data: gov } = useGovernor();
  const [killSwitchLoading, setKillSwitchLoading] = useState(false);
  const [autoSearchLoading, setAutoSearchLoading] = useState(false);
  const [budgetInput, setBudgetInput] = useState("");
  const [budgetMsg, setBudgetMsg] = useState<string | null>(null);

  const toggleKillSwitch = useCallback(async () => {
    if (!gov || killSwitchLoading) return;
    setKillSwitchLoading(true);
    try {
      await fetch(API.governorKillSwitch, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !gov.kill_switch }),
      });
    } catch {
      /* ignore */
    }
    setKillSwitchLoading(false);
  }, [gov, killSwitchLoading]);

  const toggleAutonomousSearch = useCallback(async () => {
    if (!gov || autoSearchLoading) return;
    setAutoSearchLoading(true);
    try {
      await fetch(API.governorAutonomousSearch, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !gov.autonomous_search }),
      });
    } catch {
      /* ignore */
    }
    setAutoSearchLoading(false);
  }, [gov, autoSearchLoading]);

  const updateBudget = useCallback(async () => {
    const val = parseFloat(budgetInput);
    if (isNaN(val) || val < 0) return;
    try {
      const resp = await fetch(API.governorBudget, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ceiling: val }),
      });
      if (resp.ok) {
        setBudgetMsg(`Budget updated to $${val.toFixed(2)}`);
        setBudgetInput("");
      }
    } catch {
      setBudgetMsg("Update failed");
    }
  }, [budgetInput]);

  if (!gov) {
    return (
      <div>
        <div className="dash-header">
          <h1 className="dash-title">Governor</h1>
          <div className="dash-subtitle">Loading governance state...</div>
        </div>
      </div>
    );
  }

  const budget = gov.budget;
  const spendPercent = Math.min(budget.budget_percent, 100);
  const foraging = gov.foraging;

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Governor</h1>
        <div className="dash-subtitle">
          5-layer governance stack — cost protection for autonomous agents
        </div>
      </div>

      {/* Kill Switch — Layer 5 */}
      <div className="dash-section">
        <div className="dash-section-title">Layer 5: Human Circuit Breaker</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Kill Switch</span>
            <button
              onClick={toggleKillSwitch}
              disabled={killSwitchLoading}
              style={{
                padding: "6px 16px",
                borderRadius: "var(--radius-sm)",
                border: "none",
                fontWeight: 600,
                fontSize: "13px",
                cursor: killSwitchLoading ? "not-allowed" : "pointer",
                background: gov.kill_switch ? "var(--error)" : "var(--alive)",
                color: "white",
                opacity: killSwitchLoading ? 0.5 : 1,
              }}
            >
              {gov.kill_switch
                ? "KILL SWITCH ON — All External Blocked"
                : "External Calls Active"}
            </button>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Governor Enabled</span>
            <span className="dash-row-value">{gov.enabled ? "Yes" : "No"}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Autonomous Search</span>
            <button
              onClick={toggleAutonomousSearch}
              disabled={autoSearchLoading}
              style={{
                padding: "6px 16px",
                borderRadius: "var(--radius-sm)",
                border: "none",
                fontWeight: 600,
                fontSize: "13px",
                cursor: autoSearchLoading ? "not-allowed" : "pointer",
                background: gov.autonomous_search
                  ? "var(--alive)"
                  : "var(--text-dim)",
                color: "white",
                opacity: autoSearchLoading ? 0.5 : 1,
              }}
            >
              {gov.autonomous_search
                ? "Autonomous Search ON"
                : "Autonomous Search OFF"}
            </button>
          </div>
        </div>
      </div>

      {/* Budget — Layer 4 */}
      <div className="dash-section">
        <div className="dash-section-title">Layer 4: Budget Ceiling</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Daily Spend</span>
            <span className="dash-row-value">
              ${budget.daily_spend.toFixed(4)}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Daily Ceiling</span>
            <span className="dash-row-value">
              ${budget.daily_ceiling.toFixed(2)}
            </span>
          </div>
          {/* Budget bar */}
          <div
            style={{
              margin: "8px 0",
              height: "8px",
              background: "var(--surface-3)",
              borderRadius: "4px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${spendPercent}%`,
                background:
                  spendPercent > 80
                    ? "var(--error)"
                    : spendPercent > 50
                      ? "var(--warning, orange)"
                      : "var(--alive)",
                borderRadius: "4px",
                transition: "width 0.3s ease",
              }}
            />
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Remaining</span>
            <span
              className="dash-row-value"
              style={{
                color:
                  budget.budget_remaining < 0.1 ? "var(--error)" : undefined,
              }}
            >
              ${budget.budget_remaining.toFixed(4)} (
              {(100 - spendPercent).toFixed(1)}%)
            </span>
          </div>
          {/* Update budget */}
          <div style={{ display: "flex", gap: "8px", marginTop: "8px" }}>
            <input
              type="number"
              step="0.10"
              min="0"
              value={budgetInput}
              onChange={(e) => setBudgetInput(e.target.value)}
              placeholder="New ceiling ($)"
              style={{
                flex: 1,
                background: "var(--surface-3)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 12px",
                color: "var(--text)",
                fontFamily: "inherit",
                fontSize: "13px",
                outline: "none",
              }}
            />
            <button
              onClick={updateBudget}
              disabled={!budgetInput}
              style={{
                padding: "8px 16px",
                background: "var(--accent)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                color: "white",
                fontWeight: 600,
                fontSize: "13px",
                cursor: budgetInput ? "pointer" : "not-allowed",
                opacity: budgetInput ? 1 : 0.5,
              }}
            >
              Update
            </button>
          </div>
          {budgetMsg && (
            <div
              style={{
                marginTop: "6px",
                fontSize: "12px",
                color: "var(--text-secondary)",
              }}
            >
              {budgetMsg}
            </div>
          )}
        </div>
      </div>

      {/* Rate Limits — Layer 3 */}
      <div className="dash-section">
        <div className="dash-section-title">Layer 3: Rate Limits</div>
        <div className="dash-card">
          {Object.entries(gov.rate_limits).map(([action, rl]) => {
            const pct = rl.limit > 0 ? (rl.current / rl.limit) * 100 : 0;
            const windowLabel = rl.window_seconds >= 86400 ? "/day" : "/hr";
            return (
              <div key={action} className="dash-row">
                <span
                  className="dash-row-label"
                  style={{ fontFamily: "var(--mono)", fontSize: "12px" }}
                >
                  {action}
                </span>
                <span
                  className="dash-row-value"
                  style={{
                    color: pct > 80 ? "var(--error)" : undefined,
                  }}
                >
                  {rl.current}/{rl.limit}
                  {windowLabel}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Call Counts */}
      {Object.keys(budget.call_counts).length > 0 && (
        <div className="dash-section">
          <div className="dash-section-title">Cost Breakdown (Today)</div>
          <div className="dash-card">
            {Object.entries(budget.call_counts).map(([action, count]) => (
              <div key={action} className="dash-row">
                <span
                  className="dash-row-label"
                  style={{ fontFamily: "var(--mono)", fontSize: "12px" }}
                >
                  {action}
                </span>
                <span className="dash-row-value">{count} calls</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Foraging — QA View */}
      <div className="dash-section">
        <div className="dash-section-title">
          Foraging (Autonomous Search — QA)
        </div>
        <div className="dash-card">
          {foraging && foraging.enabled !== false ? (
            <>
              <div className="dash-row">
                <span className="dash-row-label">Foraging Today</span>
                <span className="dash-row-value">
                  {foraging.forage_count ?? 0}/{foraging.max_daily ?? 30}
                </span>
              </div>
              <div className="dash-row">
                <span className="dash-row-label">Cooldown</span>
                <span className="dash-row-value">
                  {(foraging.cooldown_remaining ?? 0) > 0
                    ? `${foraging.cooldown_remaining} cycles`
                    : "Ready"}
                </span>
              </div>

              {/* Last Query & Summary — Full display */}
              {foraging.last_query && (
                <div
                  style={{
                    marginTop: "12px",
                    padding: "10px 14px",
                    background: "var(--surface-3)",
                    borderRadius: "var(--radius-sm)",
                    borderLeft: "3px solid var(--accent)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-dim)",
                      marginBottom: "4px",
                      textTransform: "uppercase",
                      letterSpacing: "0.5px",
                    }}
                  >
                    Last Query
                  </div>
                  <div style={{ fontSize: "13px", fontStyle: "italic" }}>
                    {foraging.last_query}
                  </div>
                  {foraging.last_summary && (
                    <>
                      <div
                        style={{
                          fontSize: "11px",
                          color: "var(--text-dim)",
                          marginTop: "10px",
                          marginBottom: "4px",
                          textTransform: "uppercase",
                          letterSpacing: "0.5px",
                        }}
                      >
                        Summary
                      </div>
                      <div style={{ fontSize: "13px", lineHeight: 1.5 }}>
                        {foraging.last_summary}
                      </div>
                    </>
                  )}
                </div>
              )}
              {!foraging.last_query && (
                <div
                  style={{
                    marginTop: "8px",
                    padding: "8px 12px",
                    background: "var(--surface-3)",
                    borderRadius: "var(--radius-sm)",
                    fontSize: "12px",
                    color: "var(--text-dim)",
                    lineHeight: 1.5,
                  }}
                >
                  No forage activity yet. The kernel will autonomously search
                  when it experiences boredom (flat curvature) or high
                  curiosity.
                </div>
              )}
            </>
          ) : (
            <div className="dash-row">
              <span className="dash-row-label">Status</span>
              <span
                className="dash-row-value"
                style={{ color: "var(--text-secondary)" }}
              >
                Disabled (set SEARXNG_URL to enable)
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Foraging History — Full QA trail */}
      {foraging?.history && foraging.history.length > 0 && (
        <div className="dash-section">
          <div className="dash-section-title">
            Foraging History ({foraging.history.length})
          </div>
          {foraging.history
            .slice()
            .reverse()
            .map((entry, idx) => {
              const ts = new Date(entry.timestamp * 1000);
              const timeStr = ts.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              });
              return (
                <div
                  key={idx}
                  className="dash-card"
                  style={{ marginBottom: "8px" }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: "6px",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "13px",
                        fontWeight: 600,
                        color: "var(--accent)",
                      }}
                    >
                      {entry.query}
                    </span>
                    <span
                      style={{
                        fontSize: "11px",
                        color: "var(--text-dim)",
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {timeStr} &middot; {entry.results_count} results
                    </span>
                  </div>

                  {/* Search Results */}
                  {entry.results && entry.results.length > 0 && (
                    <div style={{ marginBottom: "8px" }}>
                      {entry.results.map((r, ri) => (
                        <div
                          key={ri}
                          style={{
                            padding: "6px 10px",
                            marginBottom: "4px",
                            background: "var(--surface-3)",
                            borderRadius: "4px",
                            fontSize: "12px",
                            lineHeight: 1.4,
                          }}
                        >
                          <div style={{ fontWeight: 500 }}>{r.title}</div>
                          {r.snippet && (
                            <div
                              style={{
                                color: "var(--text-dim)",
                                marginTop: "2px",
                              }}
                            >
                              {r.snippet.length > 150
                                ? r.snippet.substring(0, 150) + "..."
                                : r.snippet}
                            </div>
                          )}
                          {r.url && (
                            <div
                              style={{
                                color: "var(--accent)",
                                fontSize: "11px",
                                marginTop: "2px",
                                opacity: 0.7,
                              }}
                            >
                              {r.url.length > 60
                                ? r.url.substring(0, 60) + "..."
                                : r.url}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Summary */}
                  {entry.summary && (
                    <div
                      style={{
                        padding: "6px 10px",
                        background: "rgba(34, 211, 238, 0.05)",
                        borderRadius: "4px",
                        borderLeft: "2px solid var(--alive)",
                        fontSize: "12px",
                        lineHeight: 1.5,
                        color: "var(--text-secondary)",
                      }}
                    >
                      <span
                        style={{
                          fontWeight: 600,
                          fontSize: "10px",
                          textTransform: "uppercase",
                          letterSpacing: "0.5px",
                        }}
                      >
                        Kernel Summary:{" "}
                      </span>
                      {entry.summary}
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      )}

      {/* Architecture Reference */}
      <div className="dash-section">
        <div className="dash-section-title">Governance Stack</div>
        <div
          className="dash-card"
          style={{
            fontFamily: "var(--mono)",
            fontSize: "12px",
            lineHeight: 1.6,
          }}
        >
          <div>L5: HUMAN CIRCUIT BREAKER — Dashboard kill switch</div>
          <div>L4: BUDGET CEILING — Hard $ cap per day</div>
          <div>L3: RATE LIMITS — Calls per window</div>
          <div>L2: INTENT GATE — Does this NEED external?</div>
          <div>L1: LOCAL-FIRST ROUTING — Ollama handles 95%</div>
        </div>
      </div>
    </div>
  );
}
