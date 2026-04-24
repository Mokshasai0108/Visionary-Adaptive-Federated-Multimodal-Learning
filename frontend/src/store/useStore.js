import { create } from 'zustand'
export const useStore = create((set) => ({
  trainingStatus: 'idle',
  currentRound: 0,
  totalRounds: 0,
  progressPct: 0,
  history: [],
  latestMetrics: null,
  sidebarOpen: true,

  trainingConfig: {
    num_rounds: 30,
    num_clients: 10,
    prompt_length: 16,
    fusion_strategy: 'dynamic',
    k_ratio: 0.3,
    batch_size: 32,
    learning_rate: 0.0001,
    lambda1: 1.0,
    lambda2: 0.5,
    seed: 42,
  },

  setTrainingState: (d) =>
    set({
      trainingStatus: d.status,
      currentRound: d.current_round,
      totalRounds: d.total_rounds,
      progressPct: d.progress_pct || (d.total_rounds > 0 ? (d.current_round / d.total_rounds) * 100 : 0),
      latestMetrics: d.current_metrics,
    }),

  setTrainingConfig: (key, value) =>
    set((state) => ({
      trainingConfig: {
        ...state.trainingConfig,
        [key]: value,
      },
    })),

  setHistory: (h) => set({ history: h }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}))
