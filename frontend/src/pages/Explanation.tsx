import { Link } from "react-router-dom";
import { usePrediction } from "../PredictionContext";
import ProbabilityBars from "../components/ProbabilityBars";
import ContributionChart from "../components/ContributionChart";

export default function Explanation() {
  const { last } = usePrediction();

  if (!last) {
    return (
      <div>
        <p className="eyebrow">Result Explanation</p>
        <h1 className="display">Nothing to explain yet</h1>
        <p className="lead">Run a prediction first, then come back to see the full
          feature contribution breakdown.</p>
        <div className="spacer" />
        <Link to="/predict" className="btn">Make a prediction →</Link>
      </div>
    );
  }

  return (
    <div>
      <p className="eyebrow">Result Explanation</p>
      <h1 className="display">{last.predicted_result}</h1>
      <p className="lead">
        {last.team_a} vs {last.team_b} · confidence {last.confidence.toLowerCase()} ·
        model: {last.model_name}
      </p>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Outcome probabilities</h3>
        <ProbabilityBars p={last} />
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Feature contributions</h3>
        <p className="muted">Green pushes toward the predicted outcome; red pushes
          against it. Values are SHAP contributions for this specific matchup.</p>
        <div className="spacer" />
        <ContributionChart data={last.contributions} />
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Top reasons</h3>
        <ol className="reasons">
          {last.reasons.map((r, i) => (
            <li key={i}><span className="num">{i + 1}</span><span>{r}</span></li>
          ))}
        </ol>
      </div>
    </div>
  );
}
