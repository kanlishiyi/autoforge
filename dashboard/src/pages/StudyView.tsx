import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  fetchStudy,
  fetchStudyImportance,
  fetchStudyTrials,
  type TrialDetail,
} from "../api";

const POLL_INTERVAL = 3000;

/* ---- dark theme colors ---- */
const C = {
  cyan: "#00f0ff",
  cyanDim: "rgba(0,240,255,0.6)",
  green: "#00ff88",
  greenDim: "rgba(0,255,136,0.6)",
  pink: "#ff2d78",
  purple: "#a855f7",
  blue: "#3b82f6",
  bgCard: "#131b2e",
  bgDeep: "#0a0e17",
  border: "#1e293b",
  borderCyan: "rgba(0,240,255,0.15)",
  textPrimary: "#e2e8f0",
  textSecondary: "#94a3b8",
  textMuted: "#64748b",
  gridLine: "rgba(0,240,255,0.06)",
};

export const StudyView: React.FC = () => {
  const { studyName } = useParams<{ studyName: string }>();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<{
    best_value: number | null;
    best_params: Record<string, unknown> | null;
    best_model_path: string | null;
    n_trials: number;
    direction: string;
  } | null>(null);
  const [importance, setImportance] = useState<Record<string, number>>({});
  const [history, setHistory] = useState<[number, number][]>([]);
  const [trials, setTrials] = useState<TrialDetail[]>([]);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadData = useCallback(
    async (isInitial = false) => {
      if (!studyName) return;
      try {
        if (isInitial) setLoading(true);
        const [s, imp, t] = await Promise.all([
          fetchStudy(studyName),
          fetchStudyImportance(studyName).catch(() => ({ study_name: studyName, importance: {} })),
          fetchStudyTrials(studyName).catch(() => null),
        ]);
        setSummary({
          best_value: s.best_value,
          best_params: s.best_params,
          best_model_path: s.best_model_path || null,
          n_trials: s.n_trials,
          direction: t?.direction || "minimize",
        });
        setHistory(s.history || []);
        setImportance(imp.importance || {});
        if (t) setTrials(t.trials);
        setError(null);
        setLastUpdate(new Date());
      } catch (e) {
        if (isInitial) setError((e as Error).message);
      } finally {
        if (isInitial) setLoading(false);
      }
    },
    [studyName]
  );

  useEffect(() => { loadData(true); }, [loadData]);
  useEffect(() => {
    if (autoRefresh) timerRef.current = setInterval(() => loadData(false), POLL_INTERVAL);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [autoRefresh, loadData]);

  /* ---- derived data ---- */
  const historyPoints = useMemo(() => history.map(([step, value]) => ({ x: step, y: value })), [history]);
  const importanceEntries = useMemo(() => Object.entries(importance), [importance]);
  const completedTrials = useMemo(() => trials.filter((t) => t.state === "completed"), [trials]);
  const failedTrials = useMemo(() => trials.filter((t) => t.state === "failed"), [trials]);
  const runningTrials = useMemo(() => trials.filter((t) => t.state === "running"), [trials]);
  const isRunning = runningTrials.length > 0 || (summary !== null && completedTrials.length + failedTrials.length < summary.n_trials && summary.n_trials > 0 && completedTrials.length > 0);

  /* ---- early returns ---- */
  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "5rem 0" }}>
        <div style={{
          display: "inline-block", width: 32, height: 32,
          border: "2px solid #1e293b", borderTopColor: "#00f0ff",
          borderRadius: "50%", animation: "spin 0.8s linear infinite",
          boxShadow: "0 0 12px rgba(0,240,255,0.2)",
        }} />
        <p style={{ marginTop: 16, color: C.textMuted, fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: 1 }}>
          LOADING STUDY...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <section>
        <div className="section-header">
          <h2>Study: {studyName}</h2>
          <Link to="/">← Back</Link>
        </div>
        <p className="error">⚠ {error}</p>
      </section>
    );
  }

  if (!summary) return <p style={{ color: C.textMuted }}>Study not found.</p>;

  /* ---- chart ---- */
  const renderHistoryChart = () => {
    if (historyPoints.length === 0) {
      return (
        <div style={{ padding: "2rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
          <p style={{ color: C.textMuted, margin: 0 }}>No optimization history available.</p>
        </div>
      );
    }

    const maxX = Math.max(...historyPoints.map((p) => p.x));
    const minX = Math.min(...historyPoints.map((p) => p.x));
    const maxY = Math.max(...historyPoints.map((p) => p.y));
    const minY = Math.min(...historyPoints.map((p) => p.y));
    const leftPad = 80, rightPad = 24, topPad = 40, bottomPad = 36;
    const width = 740, height = 320;
    const innerW = width - leftPad - rightPad;
    const innerH = height - topPad - bottomPad;

    const rangeX = maxX - minX || 1;
    const rawRangeY = maxY - minY;
    const rangeY = rawRangeY > 0 ? rawRangeY : Math.abs(maxY) * 0.01 || 1;
    const chartMinY = rawRangeY > 0 ? minY : minY - rangeY / 2;

    const scaleX = (x: number) => leftPad + ((x - minX) / rangeX) * innerW;
    const scaleY = (y: number) => height - bottomPad - ((y - chartMinY) / rangeY) * innerH;

    const pathD = historyPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ");

    // Gradient area
    const areaD = pathD + ` L ${scaleX(historyPoints[historyPoints.length - 1].x)} ${height - bottomPad} L ${scaleX(historyPoints[0].x)} ${height - bottomPad} Z`;

    const smartFormat = (v: number): string => {
      if (rangeY === 0) return v.toPrecision(6);
      const tickStep = rangeY / 4;
      const decimals = Math.max(2, Math.ceil(-Math.log10(tickStep)) + 2);
      return v.toFixed(Math.min(decimals, 10));
    };

    const yTicks = Array.from({ length: 5 }, (_, i) => {
      const val = chartMinY + (rangeY * i) / 4;
      return { val, cy: scaleY(val) };
    });

    const xLabelCount = Math.min(historyPoints.length, 6);
    const xLabels: typeof historyPoints = [];
    for (let i = 0; i < xLabelCount; i++) {
      const idx = Math.round((i / Math.max(xLabelCount - 1, 1)) * (historyPoints.length - 1));
      xLabels.push(historyPoints[idx]);
    }

    const hovered = hoveredIdx !== null ? historyPoints[hoveredIdx] : null;

    return (
      <svg
        width={width} height={height}
        style={{ background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, boxShadow: "0 4px 32px rgba(0,0,0,0.3)" }}
        onMouseLeave={() => setHoveredIdx(null)}
      >
        <defs>
          <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={C.cyan} stopOpacity="0.15" />
            <stop offset="100%" stopColor={C.cyan} stopOpacity="0" />
          </linearGradient>
          <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={C.cyan} />
            <stop offset="100%" stopColor={C.green} />
          </linearGradient>
          <filter id="glowLine">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Grid lines */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={leftPad} y1={t.cy} x2={width - rightPad} y2={t.cy} stroke={C.gridLine} />
            <text x={leftPad - 8} y={t.cy + 4} textAnchor="end" fill={C.textMuted} fontSize={10} fontFamily="var(--font-mono)">
              {smartFormat(t.val)}
            </text>
          </g>
        ))}

        {/* X axis */}
        <line x1={leftPad} y1={height - bottomPad} x2={width - rightPad} y2={height - bottomPad} stroke={C.gridLine} />

        {/* Area fill */}
        <path d={areaD} fill="url(#areaGrad)" />

        {/* Line */}
        <path d={pathD} fill="none" stroke="url(#lineGrad)" strokeWidth={2.5} filter="url(#glowLine)" />

        {/* Hover guide */}
        {hovered && (
          <line x1={scaleX(hovered.x)} y1={topPad} x2={scaleX(hovered.x)} y2={height - bottomPad} stroke={C.cyanDim} strokeDasharray="3 3" strokeWidth={1} />
        )}

        {/* Dots */}
        {historyPoints.map((p, i) => (
          <g key={i} onMouseEnter={() => setHoveredIdx(i)} style={{ cursor: "crosshair" }}>
            <circle cx={scaleX(p.x)} cy={scaleY(p.y)} r={14} fill="transparent" />
            <circle cx={scaleX(p.x)} cy={scaleY(p.y)} r={hoveredIdx === i ? 6 : 3}
              fill={hoveredIdx === i ? C.cyan : C.green}
              stroke={hoveredIdx === i ? C.bgCard : "none"} strokeWidth={2}
              style={hoveredIdx === i ? { filter: `drop-shadow(0 0 6px ${C.cyan})` } : {}}
            />
          </g>
        ))}

        {/* X labels */}
        {xLabels.map((p, i) => (
          <text key={i} x={scaleX(p.x)} y={height - 10} textAnchor="middle" fill={C.textMuted} fontSize={10} fontFamily="var(--font-mono)">
            #{p.x}
          </text>
        ))}

        {/* Title */}
        <text x={width / 2} y={22} textAnchor="middle" fill={C.textSecondary} fontSize={12} fontFamily="var(--font-mono)" fontWeight="600" letterSpacing="0.5">
          OPTIMIZATION HISTORY
        </text>

        {/* Tooltip */}
        {hovered && (() => {
          const cx = scaleX(hovered.x), cy = scaleY(hovered.y);
          const tw = 180, th = 52;
          const tx = cx + tw + 12 > width ? cx - tw - 12 : cx + 12;
          const ty = cy - th - 8 < topPad ? cy + 12 : cy - th - 8;
          return (
            <g>
              <rect x={tx} y={ty} width={tw} height={th} rx={8} fill="rgba(10,14,23,0.95)" stroke={C.borderCyan} />
              <text x={tx + 12} y={ty + 20} fill={C.textMuted} fontSize={10} fontFamily="var(--font-mono)">
                Trial #{hovered.x}
              </text>
              <text x={tx + 12} y={ty + 40} fill={C.cyan} fontSize={13} fontWeight="bold" fontFamily="var(--font-mono)">
                {smartFormat(hovered.y)}
              </text>
            </g>
          );
        })()}
      </svg>
    );
  };

  const renderImportanceChart = () => {
    if (importanceEntries.length === 0) {
      return (
        <div style={{ padding: "1.5rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
          <p style={{ color: C.textMuted, margin: 0 }}>No importance data available.</p>
        </div>
      );
    }
    return (
      <div className="importance-bars">
        {importanceEntries.map(([name, value]) => (
          <div key={name} className="importance-row">
            <span className="importance-name">{name}</span>
            <div className="importance-bar-container">
              <div className="importance-bar" style={{ width: `${Math.min(value, 1) * 100}%` }} />
            </div>
            <span className="importance-value">{value.toFixed(3)}</span>
          </div>
        ))}
      </div>
    );
  };

  /* ---- helpers ---- */
  const totalTrials = summary.n_trials;
  const doneTrials = completedTrials.length + failedTrials.length;
  const progressPct = totalTrials > 0 ? (doneTrials / totalTrials) * 100 : 0;

  const formatDuration = (d: number | null) => {
    if (d === null || d === undefined) return "—";
    if (d < 1) return `${(d * 1000).toFixed(0)}ms`;
    if (d < 60) return `${d.toFixed(1)}s`;
    return `${Math.floor(d / 60)}m ${(d % 60).toFixed(0)}s`;
  };

  const formatParamsShort = (params: Record<string, unknown>) => {
    const entries = Object.entries(params);
    if (entries.length === 0) return "—";
    return entries.map(([k, v]) => {
      const val = typeof v === "number" ? (Number.isInteger(v) ? v.toString() : v.toFixed(4)) : String(v);
      return `${k}=${val}`;
    }).join("  ");
  };

  const trialStatusStyle = (state: string) => {
    const map: Record<string, { bg: string; text: string }> = {
      completed: { bg: "rgba(0,255,136,0.08)", text: "#00ff88" },
      failed:    { bg: "rgba(255,45,120,0.08)", text: "#ff2d78" },
      running:   { bg: "rgba(0,240,255,0.08)", text: "#00f0ff" },
    };
    const s = map[state] || { bg: "rgba(100,116,139,0.08)", text: "#94a3b8" };
    return {
      display: "inline-block" as const, padding: "2px 10px", borderRadius: 20,
      fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
      background: s.bg, color: s.text, border: `1px solid ${s.text}22`,
      letterSpacing: 0.5, textTransform: "uppercase" as const,
    };
  };

  /* ---- render ---- */
  return (
    <section>
      <div className="section-header">
        <h2>Study: {studyName}</h2>
        <Link to="/">← Back</Link>
      </div>

      {/* ===== Status Bar ===== */}
      <div style={{
        display: "flex", alignItems: "center", gap: 16, marginBottom: 16,
        padding: "10px 20px", background: C.bgCard, borderRadius: 10,
        border: `1px solid ${isRunning ? "rgba(0,240,255,0.15)" : "rgba(0,255,136,0.15)"}`,
        fontSize: 12, fontFamily: "var(--font-mono)",
        boxShadow: isRunning ? "0 0 20px rgba(0,240,255,0.05)" : "0 0 20px rgba(0,255,136,0.05)",
      }}>
        <span style={{
          width: 8, height: 8, borderRadius: "50%",
          background: isRunning ? C.cyan : C.green,
          display: "inline-block",
          animation: isRunning ? "pulse 1.5s infinite" : "none",
          boxShadow: isRunning ? `0 0 8px ${C.cyan}` : `0 0 8px ${C.green}`,
        }} />
        <span style={{ fontWeight: 700, color: isRunning ? C.cyan : C.green, letterSpacing: 0.5 }}>
          {isRunning ? "RUNNING" : "COMPLETED"}
        </span>
        <span style={{ color: C.textMuted }}>
          {completedTrials.length} completed / {failedTrials.length} failed / {totalTrials} total
        </span>
        <span style={{ color: C.textMuted }}>
          direction: <span style={{ color: C.purple }}>{summary.direction}</span>
        </span>

        <label style={{
          marginLeft: "auto", display: "flex", alignItems: "center", gap: 6,
          cursor: "pointer", color: C.textMuted, fontSize: 11,
        }}>
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
          Auto-refresh ({POLL_INTERVAL / 1000}s)
        </label>
        {lastUpdate && (
          <span style={{ color: "#475569", fontSize: 10 }}>
            {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* ===== Progress Bar ===== */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.textMuted, marginBottom: 6, fontFamily: "var(--font-mono)" }}>
          <span>PROGRESS</span>
          <span>{doneTrials} / {totalTrials} ({progressPct.toFixed(0)}%)</span>
        </div>
        <div style={{ height: 4, background: "rgba(30,41,59,0.8)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{
            width: `${progressPct}%`, height: "100%",
            background: isRunning ? `linear-gradient(90deg, ${C.blue}, ${C.cyan})` : `linear-gradient(90deg, ${C.green}, ${C.cyan})`,
            borderRadius: 2, transition: "width 0.5s ease",
            boxShadow: isRunning ? `0 0 8px ${C.cyan}` : `0 0 8px ${C.green}`,
          }} />
        </div>
      </div>

      {/* ===== Metric Cards ===== */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12, marginBottom: 28 }}>
        {/* Best Value */}
        <div style={{
          padding: "16px 20px", background: C.bgCard, borderRadius: 12,
          border: `1px solid ${C.border}`, position: "relative", overflow: "hidden",
        }}>
          <div style={{
            position: "absolute", top: 0, left: 0, right: 0, height: 2,
            background: `linear-gradient(90deg, ${C.cyan}, transparent)`,
          }} />
          <div style={{ fontSize: 10, color: C.textMuted, marginBottom: 6, fontFamily: "var(--font-mono)", letterSpacing: 1, textTransform: "uppercase" }}>
            Best Value
          </div>
          <div style={{
            fontSize: 22, fontWeight: 800, fontFamily: "var(--font-mono)",
            color: summary.best_value !== null ? C.cyan : C.textMuted,
            textShadow: summary.best_value !== null ? `0 0 12px rgba(0,240,255,0.3)` : "none",
          }}>
            {summary.best_value !== null ? summary.best_value.toPrecision(8) : "—"}
          </div>
        </div>

        {/* Completed */}
        <div style={{
          padding: "16px 20px", background: C.bgCard, borderRadius: 12,
          border: `1px solid ${C.border}`, position: "relative", overflow: "hidden",
        }}>
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${C.green}, transparent)` }} />
          <div style={{ fontSize: 10, color: C.textMuted, marginBottom: 6, fontFamily: "var(--font-mono)", letterSpacing: 1, textTransform: "uppercase" }}>
            Completed
          </div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "var(--font-mono)", color: C.green, textShadow: `0 0 12px rgba(0,255,136,0.3)` }}>
            {completedTrials.length}
          </div>
        </div>

        {/* Failed */}
        <div style={{
          padding: "16px 20px", background: C.bgCard, borderRadius: 12,
          border: `1px solid ${C.border}`, position: "relative", overflow: "hidden",
        }}>
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${failedTrials.length > 0 ? C.pink : C.border}, transparent)` }} />
          <div style={{ fontSize: 10, color: C.textMuted, marginBottom: 6, fontFamily: "var(--font-mono)", letterSpacing: 1, textTransform: "uppercase" }}>
            Failed
          </div>
          <div style={{
            fontSize: 22, fontWeight: 800, fontFamily: "var(--font-mono)",
            color: failedTrials.length > 0 ? C.pink : C.textMuted,
            textShadow: failedTrials.length > 0 ? "0 0 12px rgba(255,45,120,0.3)" : "none",
          }}>
            {failedTrials.length}
          </div>
        </div>

        {/* Best Model */}
        {summary.best_model_path && (
          <div style={{
            padding: "16px 20px", background: C.bgCard, borderRadius: 12,
            border: `1px solid ${C.border}`, position: "relative", overflow: "hidden",
            gridColumn: "span 2",
          }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${C.purple}, ${C.cyan})` }} />
            <div style={{ fontSize: 10, color: C.textMuted, marginBottom: 6, fontFamily: "var(--font-mono)", letterSpacing: 1, textTransform: "uppercase" }}>
              🏆 Best Model
            </div>
            <code style={{
              fontSize: 12, fontFamily: "var(--font-mono)", color: C.greenDim,
              wordBreak: "break-all", background: "rgba(0,255,136,0.04)",
              padding: "4px 10px", borderRadius: 6, border: "1px solid rgba(0,255,136,0.1)",
              display: "inline-block",
            }}>
              {summary.best_model_path}
            </code>
          </div>
        )}
      </div>

      {/* ===== Chart ===== */}
      <h3>Optimization History</h3>
      {renderHistoryChart()}

      {/* ===== Trials Table ===== */}
      <h3 style={{ marginTop: 28 }}>Trials Detail</h3>
      {trials.length === 0 ? (
        <div style={{ padding: "2rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
          <p style={{ color: C.textMuted, margin: 0 }}>No trial data available yet.</p>
        </div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table className="table" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th style={{ width: 50 }}>#</th>
                <th style={{ width: 80 }}>State</th>
                <th style={{ width: 140 }}>Value</th>
                <th style={{ width: 90 }}>Duration</th>
                <th>Parameters</th>
              </tr>
            </thead>
            <tbody>
              {[...trials].reverse().map((t) => {
                const isBest = summary.best_value !== null && t.value !== null && t.value === summary.best_value;
                return (
                  <tr key={t.trial_id} style={{
                    background: isBest ? "rgba(0,255,136,0.04)" : t.state === "failed" ? "rgba(255,45,120,0.03)" : undefined,
                    borderLeft: isBest ? `2px solid ${C.green}` : undefined,
                  }}>
                    <td style={{ fontFamily: "var(--font-mono)", color: isBest ? C.green : C.textSecondary }}>
                      {isBest && "★ "}{t.trial_id}
                    </td>
                    <td><span style={trialStatusStyle(t.state)}>{t.state}</span></td>
                    <td style={{
                      fontFamily: "var(--font-mono)",
                      color: isBest ? C.green : t.value !== null ? C.textPrimary : C.textMuted,
                      textShadow: isBest ? `0 0 6px rgba(0,255,136,0.3)` : "none",
                    }}>
                      {t.value !== null ? t.value.toPrecision(8) : "—"}
                    </td>
                    <td style={{ fontFamily: "var(--font-mono)", color: C.textMuted, fontSize: 11 }}>
                      {formatDuration(t.duration)}
                    </td>
                    <td style={{ maxWidth: 450, wordBreak: "break-word" }}>
                      <code style={{ fontSize: 10.5, color: C.textSecondary, background: "transparent", border: "none", padding: 0 }}>
                        {formatParamsShort(t.params)}
                      </code>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ===== Best Parameters ===== */}
      <h3 style={{ marginTop: 28 }}>Best Parameters</h3>
      {!summary.best_params ? (
        <div style={{ padding: "1.5rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
          <p style={{ color: C.textMuted, margin: 0 }}>No best parameters recorded.</p>
        </div>
      ) : (
        <table className="table">
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>
            {Object.entries(summary.best_params).map(([k, v]) => (
              <tr key={k}>
                <td style={{ fontFamily: "var(--font-mono)", color: C.cyan }}>{k}</td>
                <td style={{ fontFamily: "var(--font-mono)", color: C.textPrimary }}>{String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* ===== Importance ===== */}
      <h3 style={{ marginTop: 28 }}>Parameter Importance</h3>
      {renderImportanceChart()}
    </section>
  );
};
