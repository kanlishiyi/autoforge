import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Experiment, fetchExperiments } from "../api";

const statusColors: Record<string, { bg: string; text: string; glow: string }> = {
  completed: { bg: "rgba(0,255,136,0.08)", text: "#00ff88", glow: "0 0 8px rgba(0,255,136,0.25)" },
  running:   { bg: "rgba(0,240,255,0.08)", text: "#00f0ff", glow: "0 0 8px rgba(0,240,255,0.25)" },
  created:   { bg: "rgba(168,85,247,0.08)", text: "#a855f7", glow: "0 0 8px rgba(168,85,247,0.25)" },
  failed:    { bg: "rgba(255,45,120,0.08)", text: "#ff2d78", glow: "0 0 8px rgba(255,45,120,0.25)" },
};

export const ExperimentsList: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchExperiments(statusFilter || undefined)
      .then((data) => { if (!cancelled) { setExperiments(data); setError(null); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [statusFilter]);

  return (
    <section>
      <div className="section-header">
        <h2>Experiments</h2>
        <div className="filters">
          <label>
            Status:
            <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="">All</option>
              <option value="created">Created</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
            </select>
          </label>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: "4rem 0" }}>
          <div style={{
            display: "inline-block", width: 32, height: 32,
            border: "2px solid #1e293b", borderTopColor: "#00f0ff",
            borderRadius: "50%", animation: "spin 0.8s linear infinite",
            boxShadow: "0 0 12px rgba(0,240,255,0.2)",
          }} />
          <p style={{ marginTop: 16, color: "#64748b", fontFamily: "var(--font-mono)", fontSize: 12, letterSpacing: 1 }}>
            LOADING EXPERIMENTS...
          </p>
        </div>
      )}

      {error && <p className="error">⚠ Connection failed: {error}</p>}

      {!loading && !error && experiments.length === 0 && (
        <div style={{
          textAlign: "center", padding: "4rem 2rem",
          background: "var(--bg-card)", borderRadius: 12,
          border: "1px solid var(--border-dark)",
        }}>
          <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.5 }}>⬡</div>
          <p style={{ color: "#64748b", margin: 0, fontFamily: "var(--font-mono)", fontSize: 13 }}>
            No experiments found. Run an optimization to get started.
          </p>
        </div>
      )}

      {experiments.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Status</th>
              <th>Best Value</th>
              <th>Model Path</th>
              <th>Study</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => {
              const sc = statusColors[exp.status] || { bg: "rgba(100,116,139,0.08)", text: "#94a3b8", glow: "none" };
              return (
                <tr key={exp.experiment_id}>
                  <td>
                    <Link to={`/experiments/${encodeURIComponent(exp.experiment_id)}`}>
                      {exp.experiment_id}
                    </Link>
                  </td>
                  <td style={{ color: "#e2e8f0", fontWeight: 600, fontFamily: "var(--font-mono)", fontSize: 13 }}>
                    {exp.name}
                  </td>
                  <td>
                    <span style={{
                      display: "inline-block", padding: "3px 12px", borderRadius: 20,
                      fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
                      background: sc.bg, color: sc.text, boxShadow: sc.glow,
                      border: `1px solid ${sc.text}22`, letterSpacing: 1,
                      textTransform: "uppercase",
                    }}>
                      {exp.status}
                    </span>
                  </td>
                  <td style={{
                    fontFamily: "var(--font-mono)", fontSize: 13,
                    color: exp.best_value != null ? "#00ff88" : "#475569",
                    textShadow: exp.best_value != null ? "0 0 6px rgba(0,255,136,0.3)" : "none",
                  }}>
                    {exp.best_value != null ? exp.best_value.toPrecision(8) : "—"}
                  </td>
                  <td>
                    {exp.best_model_path ? (
                      <code>{exp.best_model_path}</code>
                    ) : (
                      <span style={{ color: "#475569" }}>—</span>
                    )}
                  </td>
                  <td>
                    <Link
                      to={`/studies/${encodeURIComponent(exp.name)}`}
                      style={{
                        display: "inline-block", padding: "4px 14px", borderRadius: 6,
                        fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
                        background: "rgba(0,240,255,0.06)", color: "#00f0ff",
                        border: "1px solid rgba(0,240,255,0.15)", textDecoration: "none",
                        transition: "all 0.2s", letterSpacing: 1, textTransform: "uppercase",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(0,240,255,0.12)";
                        e.currentTarget.style.boxShadow = "0 0 16px rgba(0,240,255,0.15)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(0,240,255,0.06)";
                        e.currentTarget.style.boxShadow = "none";
                      }}
                    >
                      View →
                    </Link>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
};
