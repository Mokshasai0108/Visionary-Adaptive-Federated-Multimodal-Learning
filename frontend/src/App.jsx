import React, { useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import Header from "./components/layout/Header";
import Dashboard from "./pages/Dashboard";
import TrainingControl from "./pages/TrainingControl";
import Inference from "./pages/Inference";
import FederatedViz from "./pages/FederatedViz";
import PromptInspector from "./pages/PromptInspector";
import Results from "./pages/Results";
import Diagnostics from "./pages/Diagnostics";
import Mirror from "./pages/Mirror";
import useTrainingStore from "./store/trainingStore";

export default function App() {
  const { fetchStatus, fetchMetrics, fetchPrompts, fetchDiagnostics, fetchModelStatus, fetchModelList } = useTrainingStore();

  useEffect(() => {
    fetchStatus();
    fetchModelStatus();
    fetchModelList();
    fetchMetrics();
    fetchPrompts();
    fetchDiagnostics();
  }, []);

  return (
    <div className="flex h-screen bg-afspl-bg overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/training" element={<TrainingControl />} />
            <Route path="/inference" element={<Inference />} />
            <Route path="/mirror" element={<Mirror />} />
            <Route path="/federated" element={<FederatedViz />} />
            <Route path="/prompts" element={<PromptInspector />} />
            <Route path="/results" element={<Results />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
