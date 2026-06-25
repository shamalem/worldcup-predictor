import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { TeamsResponse, PredictResponse, ScorePredictResponse } from "../types";
import { usePrediction } from "../PredictionContext";
import ProbabilityBars from "../components/ProbabilityBars";

export default function Predict() {
  const { setLast } = usePrediction();
  const navigate = useNavigate();

  const [catalog, setCatalog] = useState<TeamsResponse | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [year, setYear] = useState<number>(2014);
  const [stage, setStage] = useState("Group Stage");
  const [neutral, setNeutral] = useState(true);
  const [host, setHost] = useState<"A" | "B" | "none">("none");

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [score, setScore] = useState<ScorePredictResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.teams()
      .then((c) => {
        setCatalog(c);
        setTeamA(c.teams[0] ?? "");
        setTeamB(c.teams[1] ?? "");
        if (c.years.length) setYear(c.years[c.years.length - 1]);
      })
      .catch((e) => setLoadError(e.message));
  }, []);

  async function submit() {
    setErr(null);
    if (teamA === teamB) { setErr("Pick two different teams."); return; }
    setBusy(true);
    try {
      // Outcome (win/draw/loss) and scoreline are independent models; fetch both.
      const [res, sc] = await Promise.all([
        api.predict({ team_a: teamA, team_b: teamB, year, stage, neutral, host }),
        api.predictScore({ team_a: teamA, team_b: teamB, neutral, host }),
      ]);
      setResult(res);
      setScore(sc);
      setLast(res);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (loadError) {
    return (
      <div className="card error">
        Couldn't load teams: {loadError}.<br />
        Make sure the backend is running and you've trained the model
        (<span className="mono">python -m ml.train</span>).
      </div>
    );
  }
  if (!catalog) return <p className="muted">Loading teams…</p>;

  return (
    <div>
      <p className="eyebrow">Predict Match</p>
      <h1 className="display">Set the fixture</h1>

      <div className="card">
        <div className="grid cols-2">
          <div>
            <label>Team A</label>
            <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
              {catalog.teams.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label>Team B</label>
            <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
              {catalog.teams.map((t) => <option key={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label>World Cup year</label>
            <select value={year} onChange={(e) => setYear(Number(e.target.value))}>
              {catalog.years.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <div>
            <label>Stage</label>
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              {catalog.stages.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div className="spacer" />
        <div className="toggle-row">
          <label className="switch">
            <input type="checkbox" checked={neutral} onChange={(e) => setNeutral(e.target.checked)} />
            Neutral venue
          </label>
          <div>
            <label>Host nation</label>
            <select value={host} onChange={(e) => setHost(e.target.value as "A" | "B" | "none")}>
              <option value="none">No host advantage</option>
              <option value="A">Team A is host</option>
              <option value="B">Team B is host</option>
            </select>
          </div>
        </div>

        <div className="spacer" />
        <button className="btn" disabled={busy} onClick={submit}>
          {busy ? "Crunching…" : "Predict outcome & score"}
        </button>
        {err && <><div className="spacer" /><div className="error">{err}</div></>}
      </div>

      {result && (
        <>
          <div className="spacer" />
          <div className="scoreboard">
            <div className="matchup">
              <span className="team-name">{result.team_a}</span>
              <span className="vs">vs</span>
              <span className="team-name">{result.team_b}</span>
            </div>
            <div className="verdict">{result.predicted_result}</div>
            <span className="confidence-pill">Confidence: {result.confidence} · {result.model_name}</span>
            <ProbabilityBars p={result} />
          </div>
        </>
      )}

      {score && (
        <>
          <div className="spacer" />
          <div className="card score-card">
            <h3 className="section-title">Most likely scoreline</h3>
            <div className="score-hero">
              <span className="score-team">{score.team_a}</span>
              <span className="score-digits">
                {score.most_likely_score.team_a}
                <span className="score-dash">–</span>
                {score.most_likely_score.team_b}
              </span>
              <span className="score-team">{score.team_b}</span>
            </div>
            <div className="score-sub">
              {Math.round(score.most_likely_score.probability * 100)}% likely · expected goals{" "}
              {score.expected_goals.team_a} – {score.expected_goals.team_b}
            </div>

            <div className="spacer" />
            <h4 className="section-title small">Top scorelines</h4>
            <div className="scoreline-list">
              {score.top_scorelines.map((s) => {
                const pct = Math.round(s.probability * 100);
                const isTop =
                  s.team_a === score.most_likely_score.team_a &&
                  s.team_b === score.most_likely_score.team_b;
                return (
                  <div className={"scoreline-row" + (isTop ? " top" : "")} key={s.label}>
                    <span className="scoreline-label">{s.label}</span>
                    <span className="scoreline-bar">
                      <span className="scoreline-fill" style={{ width: `${Math.max(pct * 4, 6)}%` }} />
                    </span>
                    <span className="scoreline-pct">{pct}%</span>
                  </div>
                );
              })}
            </div>

            <div className="spacer" />
            <p className="muted small">
              Scoreline model: {score.model_name}. Exact scores are inherently hard to
              call — treat these as a probability ranking, not a certainty.
            </p>
          </div>
        </>
      )}

      {result && (
        <>
          <div className="spacer" />
          <div className="card">
            <h3 className="section-title">Why this prediction</h3>
            <ol className="reasons">
              {result.reasons.map((r, i) => (
                <li key={i}><span className="num">{i + 1}</span><span>{r}</span></li>
              ))}
            </ol>
            <div className="spacer" />
            <button className="btn ghost" onClick={() => navigate("/explanation")}>
              See full feature breakdown →
            </button>
          </div>
        </>
      )}
    </div>
  );
}
