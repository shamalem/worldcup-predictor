# ⚽ Explainable FIFA World Cup Match Predictor

Predict the outcome of a **FIFA World Cup** match — **Team A win / Draw / Team B win** —
with calibrated probabilities and **model-based explanations** (SHAP + template
sentences, **no LLM**).

> Scope is intentionally limited to **World Cup finals matches only**. It is not a
> general predictor for leagues, friendlies, the Euros, or the Champions League.

```
React + TypeScript (Vite)  ──►  FastAPI  ──►  scikit-learn / XGBoost + SHAP
                                   │
                                   └──►  PostgreSQL (or SQLite)
```

---

## ✨ Features

- **World-Cup-specific feature engineering** — historical win rate, goals for/against,
  goal difference, match & knockout experience, head-to-head, host advantage, stage
  importance and era. All features are computed **only from matches before the fixture**
  (no data leakage), and the training set is mirrored so the model is symmetric in
  Team A / Team B.
- **Model bake-off** — Logistic Regression, Random Forest and XGBoost are trained and
  compared; the best one (lowest validation log loss) is served.
- **Exact-scoreline prediction** — a separate Poisson + Dixon-Coles model estimates each
  team's expected goals and turns that into a probability for every scoreline (most likely
  score, top-5 scorelines, and a win/draw/loss split). Trained by the same `ml.train`
  command, served at `POST /api/predict-score`.
- **Explainability** — SHAP contributions per prediction, turned into readable reasons
  and a feature-contribution chart.
- **Production shape** — FastAPI backend, typed React frontend, Postgres/SQLite, Docker
  Compose, GitHub Actions CI, unit tests.

---

## 🚀 Quick start (local, SQLite — no Docker)

You need **Python 3.12+** and **Node 20+**.

> **A trained model is already included** in `backend/artifacts/` (LogisticRegression,
> trained on 1,012 real World Cup finals matches, 1930–2026), so you can skip straight to
> step 5 and run the API immediately. Steps 2–4 are only needed if you want to
> **re-train from scratch** (e.g. after the dataset updates) — and re-training with
> `xgboost`/`shap` installed will use the full XGBoost + SHAP path instead of the
> scikit-learn fallbacks. The bundled artifacts are git-ignored, so they won't be pushed
> to GitHub; each clone re-generates them with steps 2–4.

```bash
# 1. Backend setup
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 2. Get the dataset (downloads data/results.csv)
#    NOTE: the martj42 CSVs are already bundled in data/ in this download,
#    so you can SKIP this step. Run it only to refresh to the latest data.
python -m scripts.download_data
#    If you have no internet, the bundled data/results.csv is already in place.

# 3. Train + compare models (creates backend/artifacts/)
#    A pre-trained model is already bundled in backend/artifacts/, so the API
#    runs out of the box. Run this to do the full Logistic/RandomForest/XGBoost
#    bake-off yourself (XGBoost is used when installed; the trainer falls back
#    to GradientBoosting otherwise).
python -m ml.train

# 4. Seed the database (optional but recommended)
python -m scripts.seed_db

# 5. Run the API
uvicorn app.main:app --reload
#    -> http://localhost:8000/docs
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
#    -> http://localhost:5173
```

Open **http://localhost:5173**, go to **Predict**, and try e.g. *Brazil vs Germany, 2014, Final*.

---

## 🐙 Put this on your own GitHub

This folder is a ready-to-push repository (it already has `.gitignore`, CI, and a
license). From inside `worldcup-predictor/`:

```bash
git init
git add .
git commit -m "Initial commit: Explainable World Cup match predictor"

# Create an EMPTY repo on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/worldcup-predictor.git
git branch -M main
git push -u origin main
```

After that, on any computer you can `git clone` it back and follow **Quick start** above.
The dataset and trained model are git-ignored on purpose, so each machine runs
`download_data` + `ml.train` once to regenerate them.

---

## 🐳 Run everything with Docker

```bash
# Train the model first so artifacts exist (mounted into the backend container):
cd backend
pip install -r requirements.txt
python -m scripts.download_data
python -m ml.train
cd ..

# Then bring up Postgres + backend + frontend:
docker compose up --build
```

- Frontend: **http://localhost:8080**
- Backend API + docs: **http://localhost:8000/docs**
- Postgres: localhost:5432 (`worldcup` / `worldcup`)

---

## ☁️ Deploy to Render (one public URL, free tier)

This repo ships a `render.yaml` blueprint and a root `Dockerfile` that build the
frontend and backend into **one web service** — the API and the website are served
from the same origin (no CORS setup), and the model is downloaded + trained during
the build, so the running container starts ready to serve.

1. Push this repo to GitHub (you've already done this).
2. Go to **[render.com](https://render.com)** → sign in with GitHub.
3. Click **New + → Blueprint**, pick your `worldcup-predictor` repo, and **Apply**.
   Render reads `render.yaml`, builds the Docker image, and deploys it.
4. Wait for the first build (a few minutes — it installs deps and trains the model).
   When it's live, open the service's URL (like `https://worldcup-predictor.onrender.com`)
   and use the app directly. The API docs are at `<your-url>/docs`.

Notes:
- **Free tier** spins the service down after inactivity, so the first request after a
  while takes ~30–60s to wake up. That's normal.
- The build trains the model automatically; you don't run any commands.
- Predictions are stored in SQLite on the container's ephemeral disk, so they reset on
  each deploy. To keep them, create a Render **PostgreSQL** instance and set the
  `DATABASE_URL` env var to its Internal URL (see the comment in `render.yaml`).

You can also deploy the same `Dockerfile` to any Docker host:

```bash
docker build -t worldcup-predictor .
docker run -p 8000:8000 worldcup-predictor
# open http://localhost:8000
```

---

## 🔌 API

| Method | Endpoint                 | Description                                   |
|--------|--------------------------|-----------------------------------------------|
| GET    | `/api/health`            | Service + model status                        |
| GET    | `/api/teams`             | Available teams, years, stages                |
| POST   | `/api/predict`           | Predict a match (probabilities + explanation) |
| POST   | `/api/predict-score`     | Predict the exact scoreline distribution      |
| GET    | `/api/model-performance` | Best model, metrics, feature importance       |

**Example request**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"team_a":"Brazil","team_b":"Germany","year":2014,"stage":"Final","neutral":true,"host":"none"}'
```

**Example response (abridged)**

```json
{
  "predicted_result": "Brazil win",
  "probabilities": { "team_a": 0.58, "draw": 0.24, "team_b": 0.18 },
  "confidence": "High",
  "reasons": [
    "Brazil has a stronger historical World Cup win rate.",
    "Brazil has a better average goal difference in WC matches.",
    "Germany has weaker performance in similar knockout-stage matches.",
    "Brazil is more experienced at the World Cup."
  ],
  "contributions": [ { "feature": "a_win_rate", "contribution": 0.11 }, ... ],
  "model_name": "XGBoost"
}
```

---

## 🗂️ Project structure

```
worldcup-predictor/
├── backend/
│   ├── app/                 # FastAPI app
│   │   ├── main.py          # entrypoint, CORS, startup
│   │   ├── routers/         # /predict /teams /model-performance /health
│   │   ├── prediction.py    # model service (load + predict + explain)
│   │   ├── database.py      # SQLAlchemy (SQLite/Postgres)
│   │   ├── models_db.py     # teams, matches, predictions, model metadata
│   │   └── schemas.py       # Pydantic contracts
│   ├── ml/                  # the ML pipeline
│   │   ├── data_prep.py     # load + filter WC matches
│   │   ├── features.py      # leakage-free feature engineering
│   │   ├── train.py         # train/compare models, save artifacts
│   │   └── explain.py       # SHAP -> template explanations
│   ├── scripts/             # download_data, seed_db
│   ├── tests/               # pytest unit tests
│   └── Dockerfile
├── frontend/                # React + TypeScript (Vite)
│   ├── src/pages/           # Home, Predict, Explanation, Performance, About
│   ├── src/components/      # Navbar, ProbabilityBars, ContributionChart
│   └── Dockerfile           # multi-stage build -> nginx
├── docker-compose.yml
├── .github/workflows/ci.yml
└── data/                    # results.csv lives here (git-ignored)
```

---

## 🧪 Tests

```bash
cd backend
pytest -q
```

Covers feature engineering (including a no-leakage check), the explanation layer, and
API smoke tests via FastAPI's `TestClient`.

---

## ⚠️ Honest limitations

- The public dataset labels the **tournament** (`FIFA World Cup`) but not the exact
  knockout round of each historical match, so `stage` is treated as a **contextual input
  you provide** rather than mined per match. If you enrich `results.csv` with a `stage`
  column, `data_prep.py` will pick it up automatically.
- Predictions reflect **long-run historical tendencies**, not live squad form, injuries,
  or tactics. Treat them as an explainable statistical baseline.

## 📊 Data

International football results dataset by **martj42**
(GitHub: `martj42/international_results`, also on Kaggle). Gathered from Wikipedia,
rsssf.com and football associations. Not redistributed here — fetch it with
`scripts/download_data.py`.

## 📄 License

MIT — see [LICENSE](./LICENSE).
