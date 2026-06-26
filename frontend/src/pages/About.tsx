export default function About() {
  return (
    <div>
      <p className="eyebrow">About</p>
      <h1 className="display">How it works</h1>
      <p className="lead">An end-to-end, explainable ML system scoped strictly to FIFA
        World Cup finals matches.</p>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Pipeline</h3>
        <ol className="reasons">
          <li><span className="num">1</span><span><b>Data</b> — international results filtered to the <span className="mono">FIFA World Cup</span> tournament (no qualifiers, no friendlies).</span></li>
          <li><span className="num">2</span><span><b>Features</b> — per-team win rate, goals for/against, goal difference, match &amp; knockout experience, head-to-head, host advantage, stage and era — all computed from matches before the fixture.</span></li>
          <li><span className="num">3</span><span><b>Team strength (Elo)</b> — an importance-weighted Elo rating computed over <i>all</i> ~49,000 internationals (not just World Cup games), so even teams with few WC matches get a strong, current strength estimate. The rating gap is the model's most influential feature.</span></li>
          <li><span className="num">4</span><span><b>Models</b> — Logistic Regression, Random Forest and XGBoost are trained and compared; the best by validation log loss is served.</span></li>
          <li><span className="num">5</span><span><b>Explainability</b> — SHAP contributions per prediction, converted to template sentences. No LLM involved.</span></li>
          <li><span className="num">6</span><span><b>Scoreline model</b> — a separate Poisson + Dixon-Coles model estimates each side's expected goals and produces a probability for every scoreline, giving the most likely exact score alongside the win/draw/loss call.</span></li>
        </ol>
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Stack</h3>
        <p className="muted">FastAPI · scikit-learn · XGBoost · SHAP · PostgreSQL/SQLite ·
          React + TypeScript · Vite · Docker · GitHub Actions.</p>
      </div>

      <div className="spacer" />
      <div className="card">
        <h3 className="section-title">Honest limits</h3>
        <p className="muted">The public dataset labels the tournament but not the exact
          knockout round per historical match, so stage is treated as a contextual input
          you provide rather than mined per-match. Predictions reflect long-run historical
          tendencies, not live squad form or injuries.</p>
      </div>
    </div>
  );
}
