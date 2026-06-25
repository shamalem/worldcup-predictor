export interface TeamsResponse {
  teams: string[];
  years: number[];
  stages: string[];
}

export interface FeatureContribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
}

export interface PredictRequest {
  team_a: string;
  team_b: string;
  year: number;
  stage: string;
  neutral: boolean;
  host: "A" | "B" | "none";
}

export interface PredictResponse {
  team_a: string;
  team_b: string;
  predicted_result: string;
  probabilities: { team_a: number; draw: number; team_b: number };
  confidence: string;
  reasons: string[];
  contributions: FeatureContribution[];
  model_name: string;
}

export interface ModelPerformance {
  best_model: string;
  n_matches: number;
  metrics: Record<string, { accuracy: number; f1_macro: number; log_loss: number }>;
  feature_importance: Record<string, number>;
  feature_display: Record<string, string>;
  class_labels: string[];
}
