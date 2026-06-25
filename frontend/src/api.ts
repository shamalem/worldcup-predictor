import type {
  TeamsResponse,
  PredictRequest,
  PredictResponse,
  ModelPerformance,
  ScorePredictRequest,
  ScorePredictResponse,
} from "./types";

const BASE = "/api";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${BASE}/health`).then((r) => handle<{ status: string; model_loaded: boolean }>(r)),
  teams: () => fetch(`${BASE}/teams`).then((r) => handle<TeamsResponse>(r)),
  performance: () => fetch(`${BASE}/model-performance`).then((r) => handle<ModelPerformance>(r)),
  predict: (body: PredictRequest) =>
    fetch(`${BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handle<PredictResponse>(r)),
  predictScore: (body: ScorePredictRequest) =>
    fetch(`${BASE}/predict-score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => handle<ScorePredictResponse>(r)),
};
