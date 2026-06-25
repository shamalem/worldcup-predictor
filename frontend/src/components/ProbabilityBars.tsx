import type { PredictResponse } from "../types";

export default function ProbabilityBars({ p }: { p: PredictResponse }) {
  const rows = [
    { key: "a", label: p.team_a, val: p.probabilities.team_a, cls: "fill-a" },
    { key: "d", label: "Draw", val: p.probabilities.draw, cls: "fill-draw" },
    { key: "b", label: p.team_b, val: p.probabilities.team_b, cls: "fill-b" },
  ];
  return (
    <div className="prob-bars">
      {rows.map((r) => (
        <div className="prob-row" key={r.key}>
          <span className="prob-label">{r.label}</span>
          <span className="prob-track">
            <span className={`prob-fill ${r.cls}`} style={{ width: `${(r.val * 100).toFixed(1)}%` }} />
          </span>
          <span className="prob-val">{(r.val * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}
