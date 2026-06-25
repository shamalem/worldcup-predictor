import { useEffect, useState } from "react";
import { api } from "../api";
import type { ModelPerformance } from "../types";

export default function Performance() {
  const [perf, setPerf] = useState<ModelPerformance | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.performance().then(setPerf).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="card error">Couldn't load performance: {err}</div>;
  if (!perf) return <p className="muted">Loading…</p>;

  const importance = Object.entries(perf.feature_importance).slice(0, 12);

  return (
    <div>
      <p className="eyebrow">Model Performance</p>
      <h1 className="display">Under the hood</h1>

      <div className="kpis">
        <div className="kpi"><div className="v">{perf.best_model}</div><div className="l">Best model</div></div>
        <div className="kpi"><div className="v">{perf.n_matches.toLocaleString()}</div><div className="l">WC matches</div></div>
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Validation metrics</h3>
        <table>
          <thead><tr><th>Model</th><th className="num">Accuracy</th><th className="num">F1 (macro)</th><th className="num">Log loss</th></tr></thead>
          <tbody>
            {Object.entries(perf.metrics).map(([name, m]) => (
              <tr key={name} className={name === perf.best_model ? "winner" : ""}>
                <td>{name}</td>
                <td className="mono">{m.accuracy.toFixed(3)}</td>
                <td className="mono">{m.f1_macro.toFixed(3)}</td>
                <td className="mono">{m.log_loss.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ marginTop: 14 }}>
          Winner chosen by lowest validation log loss — calibrated probabilities matter
          most for an explainable predictor.
        </p>
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Global feature importance</h3>
        <table>
          <thead><tr><th>Feature</th><th className="num">Importance</th></tr></thead>
          <tbody>
            {importance.map(([f, v]) => (
              <tr key={f}>
                <td>{perf.feature_display[f] ?? f}</td>
                <td className="mono">{(v * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
