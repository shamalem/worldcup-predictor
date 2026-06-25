import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div>
      <p className="eyebrow">Explainable ML · FIFA World Cup only</p>
      <h1 className="display">Who wins the match before<br />it's even played?</h1>
      <p className="lead">
        Pick two nations, a World Cup year, and the stage. A gradient-boosted model
        trained on every World Cup finals match since 1930 returns win / draw / loss
        probabilities &mdash; and tells you, in plain language, exactly which factors
        drove the call. No black box, no LLM guesswork.
      </p>
      <div className="spacer" />
      <div className="toggle-row">
        <Link to="/predict" className="btn">Predict a match →</Link>
        <Link to="/performance" className="btn ghost">See model performance</Link>
      </div>
      <div className="spacer" />
      <div className="grid cols-2">
        <div className="card">
          <h3 className="section-title">Built on real history</h3>
          <p className="muted">Filtered strictly to FIFA World Cup finals matches.
            Team features &mdash; win rate, goal difference, knockout experience,
            head-to-head &mdash; are computed only from matches played <em>before</em> the
            fixture, so there's no leakage.</p>
        </div>
        <div className="card">
          <h3 className="section-title">Explanations you can read</h3>
          <p className="muted">Every prediction ships with a SHAP-driven feature
            contribution chart and template-based sentences like &ldquo;Brazil has a
            stronger historical World Cup win rate.&rdquo;</p>
        </div>
      </div>
    </div>
  );
}
