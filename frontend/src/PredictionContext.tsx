import { createContext, useContext, useState, ReactNode } from "react";
import type { PredictResponse } from "./types";

interface Ctx {
  last: PredictResponse | null;
  setLast: (p: PredictResponse | null) => void;
}

const PredictionCtx = createContext<Ctx>({ last: null, setLast: () => {} });

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [last, setLast] = useState<PredictResponse | null>(null);
  return <PredictionCtx.Provider value={{ last, setLast }}>{children}</PredictionCtx.Provider>;
}

export const usePrediction = () => useContext(PredictionCtx);
