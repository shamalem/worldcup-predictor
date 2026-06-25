import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { PredictionProvider } from "./PredictionContext";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <PredictionProvider>
        <App />
      </PredictionProvider>
    </BrowserRouter>
  </React.StrictMode>
);
