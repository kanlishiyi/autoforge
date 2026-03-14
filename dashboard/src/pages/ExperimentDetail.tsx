import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Experiment, MetricRecord, fetchExperiment, fetchMetrics } from "../api";

const C = {
  cyan: "#00f0ff",
  green: "#00ff88",
  bgCard: "#131b2e",
  border: "#1e293b",
  gridLine: "rgba(0,240,255,0.06)",
  textPrimary: "#e2e8f0",
  textSecondary: "#94a3b8",
  textMuted: "#64748b",
};

interface SeriesPoint { x: number; y: number; }

const MetricChart: React.FC<{ points: SeriesPoint[]; metricName: string }> = ({ points, metricName }) => {
  if (points.length === 0) {
    return (
      <div style={{ padding: "2rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
        <p style={{ color: C.textMuted, margin: 0 }}>No data for this metric.</p>
      </div>
    );
  }

  const maxX = Math.max(...points.map((p) => p.x));
  const maxY = Math.max(...points.map((p) => p.y));
  const minY = Math.min(...points.map((p) => p.y));
  const leftPad = 72, rightPad = 20, topPad = 36, bottomPad = 32;
  const width = 660, height = 280;
  const innerW = width - leftPad - rightPad;
  const innerH = height - topPad - bottomPad;

  const rangeY = maxY - minY || 1;
  const scaleX = (x: number) => leftPad + (x / Math.max(maxX || 1, 1)) * innerW;
  const scaleY = (y: number) => height - bottomPad - ((y - minY) / rangeY) * innerH;

  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ");
  const areaD = pathD + ` L ${scaleX(points[points.length - 1].x)} ${height - bottomPad} L ${scaleX(points[0].x)} ${height - bottomPad} Z`;

  const yTicks = Array.from({ length: 5 }, (_, i) => {
    const val = minY + (rangeY * i) / 4;
    return { val, cy: scaleY(val) };
  });

  return (
    <svg width={width} height={height}
      style={{ background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, boxShadow: "0 4px 24px rgba(0,0,0,0.2)" }}
    >
      <defs>
        <linearGradient id="metricArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={C.cyan} stopOpacity="0.12" />
          <stop offset="100%" stopColor={C.cyan} stopOpacity="0" />
        </linearGradient>
        <filter id="metricGlow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={leftPad} y1={t.cy} x2={width - rightPad} y2={t.cy} stroke={C.gridLine} />
          <text x={leftPad - 6} y={t.cy + 4} textAnchor="end" fill={C.textMuted} fontSize={10} fontFamily="var(--font-mono)">
            {t.val.toPrecision(4)}
          </text>
        </g>
      ))}

      <path d={areaD} fill="url(#metricArea)" />
      <path d={pathD} fill="none" stroke={C.cyan} strokeWidth={2} filter="url(#metricGlow)" />

      {points.map((p, i) => (
        <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={3} fill={C.cyan}
          style={{ filter: `drop-shadow(0 0 3px ${C.cyan})` }}
        />
      ))}

      <text x={width / 2} y={20} textAnchor="middle" fill={C.textSecondary} fontSize={12} fontFamily="var(--font-mono)" fontWeight="600" letterSpacing="0.5">
        {metricName.toUpperCase()}
      </text>
    </svg>
  );
};

export const ExperimentDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [metrics, setMetrics] = useState<MetricRecord[]>([]);
  const [metricName, setMetricName] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const [exp, metricsRes] = await Promise.all([fetchExperiment(id), fetchMetrics(id)]);
        if (!cancelled) { setExperiment(exp); setMetrics(metricsRes.metrics); setError(null); }
      } catch (e) { if (!cancelled) setError((e as Error).message); }
      finally { if (!cancelled) setLoading(false); }
    }
    load();
    return () => { cancelled = true; };
  }, [id]);

  const metricNames = useMemo(() => Array.from(new Set(metrics.map((m) => m.metric_name))), [metrics]);

  useEffect(() => {
    if (!metricName && metricNames.length > 0) setMetricName(metricNames[0]);
  }, [metricName, metricNames]);

  const currentSeries: SeriesPoint[] = useMemo(
    () => metrics.filter((m) => m.metric_name === metricName).sort((a, b) => a.step - b.step).map((m) => ({ x: m.step, y: m.value })),
    [metrics, metricName]
  );

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "5rem 0" }}>
        <div style={{
          display: "inline-block", width: 32, height: 32,
          border: "2px solid #1e293b", borderTopColor: "#00f0ff",
          borderRadius: "50%", animation: "spin 0.8s linear infinite",
        }} />
        <p style={{ marginTop: 16, color: C.textMuted, fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: 1 }}>
          LOADING...
        </p>
      </div>
    );
  }

  if (error) return <p className="error">⚠ Failed to load experiment: {error}</p>;
  if (!experiment) return <p style={{ color: C.textMuted }}>Experiment not found.</p>;

  const statusMap: Record<string, { color: string; bg: string }> = {
    completed: { color: "#00ff88", bg: "rgba(0,255,136,0.08)" },
    running:   { color: "#00f0ff", bg: "rgba(0,240,255,0.08)" },
    created:   { color: "#a855f7", bg: "rgba(168,85,247,0.08)" },
    failed:    { color: "#ff2d78", bg: "rgba(255,45,120,0.08)" },
  };
  const sc = statusMap[experiment.status] || { color: "#94a3b8", bg: "rgba(100,116,139,0.08)" };

  return (
    <section>
      <div className="section-header">
        <h2>Experiment: {experiment.name}</h2>
        <Link to="/">← Back</Link>
      </div>

      {/* Meta cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginBottom: 28 }}>
        <div style={{ padding: "14px 20px", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 10, color: C.textMuted, fontFamily: "var(--font-mono)", letterSpacing: 1, marginBottom: 6 }}>ID</div>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: C.textPrimary, wordBreak: "break-all" }}>{experiment.experiment_id}</div>
        </div>
        <div style={{ padding: "14px 20px", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 10, color: C.textMuted, fontFamily: "var(--font-mono)", letterSpacing: 1, marginBottom: 6 }}>STATUS</div>
          <span style={{
            display: "inline-block", padding: "3px 12px", borderRadius: 20,
            fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
            background: sc.bg, color: sc.color, border: `1px solid ${sc.color}22`,
            letterSpacing: 1, textTransform: "uppercase",
          }}>
            {experiment.status}
          </span>
        </div>
        <div style={{ padding: "14px 20px", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}` }}>
          <div style={{ fontSize: 10, color: C.textMuted, fontFamily: "var(--font-mono)", letterSpacing: 1, marginBottom: 6 }}>BEST VALUE</div>
          <div style={{
            fontFamily: "var(--font-mono)", fontSize: 18, fontWeight: 800,
            color: experiment.best_value != null ? C.cyan : C.textMuted,
            textShadow: experiment.best_value != null ? `0 0 10px rgba(0,240,255,0.3)` : "none",
          }}>
            {experiment.best_value ?? "—"}
          </div>
        </div>
      </div>

      <h3>Metrics</h3>
      {metricNames.length === 0 ? (
        <div style={{ padding: "2rem", background: C.bgCard, borderRadius: 12, border: `1px solid ${C.border}`, textAlign: "center" }}>
          <p style={{ color: C.textMuted, margin: 0 }}>No metrics recorded yet.</p>
        </div>
      ) : (
        <>
          <div className="filters" style={{ marginBottom: 16 }}>
            <label>
              Metric:
              <select value={metricName} onChange={(e) => setMetricName(e.target.value)}>
                {metricNames.map((name) => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>
          </div>
          <MetricChart points={currentSeries} metricName={metricName} />
        </>
      )}
    </section>
  );
};
