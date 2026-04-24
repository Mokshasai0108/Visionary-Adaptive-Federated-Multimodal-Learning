import { create } from "zustand";
import { trainAPI, modelAPI, metricsAPI, promptsAPI, diagnosticsAPI } from "../services/api";

const useTrainingStore = create((set, get) => ({
  // Training state
  status: "idle",
  isTraining: false,
  currentRound: 0,
  totalRounds: 0,
  bestMetric: 0,
  currentMetrics: {},
  totalCommBytes: 0,
  trainingHistory: [],
  comparisonTable: [],

  // Model status
  modelLoaded: false,
  modelRound: 0,
  modelVersion: "unknown",
  modelName: "default",
  availableModels: [],

  // Prompt state
  promptState: null,
  lastFusionWeights: [0.33, 0.33, 0.34],

  // Diagnostics
  diagnostics: null,

  // UI state
  pollingInterval: null,
  error: null,

  startTraining: async (config) => {
    try {
      await trainAPI.start(config);
      get().startPolling();
    } catch (err) {
      set({ error: err.response?.data?.detail || "Failed to start training" });
    }
  },

  stopTraining: async () => {
    await trainAPI.stop();
    get().stopPolling();
  },

  fetchStatus: async () => {
    try {
      const { data } = await trainAPI.status();
      set({
        status: data.status,
        isTraining: data.is_training,
        currentRound: data.current_round,
        totalRounds: data.total_rounds,
        bestMetric: data.best_metric,
        currentMetrics: data.current_metrics || {},
        totalCommBytes: data.total_comm_bytes || 0,
      });
      if (data.is_training && !get().pollingInterval) {
        get().startPolling();
      }
    } catch {}
  },

  fetchMetrics: async () => {
    try {
      const { data } = await metricsAPI.get();
      set({
        trainingHistory: data.training_history || [],
        comparisonTable: data.comparison_table || [],
      });
    } catch {}
  },

  fetchModelStatus: async () => {
    try {
      const { data } = await modelAPI.status();
      set({
        modelLoaded: data.loaded,
        modelRound: data.round,
        modelVersion: data.version,
        modelName: data.model_name || "default",
      });
    } catch {}
  },

  fetchModelList: async () => {
    try {
      const { data } = await modelAPI.list();
      set({ availableModels: data.models || [] });
    } catch {}
  },

  switchModel: async (modelName) => {
    try {
      await modelAPI.switch(modelName);
      await get().fetchModelStatus();
      await get().fetchModelList();
    } catch (err) {
      set({ error: err.response?.data?.detail || `Failed to switch to ${modelName}` });
    }
  },

  fetchPrompts: async () => {
    try {
      const { data } = await promptsAPI.get();
      set({
        promptState: data,
        lastFusionWeights: data.last_fusion_weights || [0.33, 0.33, 0.34],
      });
    } catch {}
  },

  fetchDiagnostics: async () => {
    try {
      const { data } = await diagnosticsAPI.get();
      set({ diagnostics: data });
    } catch {}
  },

  startPolling: () => {
    const id = setInterval(async () => {
      await get().fetchStatus();
      await get().fetchModelStatus();
      await get().fetchModelList();
      await get().fetchMetrics();
      await get().fetchPrompts();
      await get().fetchDiagnostics();
      if (!get().isTraining) get().stopPolling();
    }, 2000);
    set({ pollingInterval: id });
  },

  stopPolling: () => {
    const id = get().pollingInterval;
    if (id) clearInterval(id);
    set({ pollingInterval: null });
  },

  clearError: () => set({ error: null }),
}));

export default useTrainingStore;
