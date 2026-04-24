import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Cpu, Network, Zap, TrendingUp, Database } from 'lucide-react'
import Header from '../components/layout/Header'
import StatCard from '../components/panels/StatCard'
import LossChart from '../components/charts/LossChart'
import MetricChart from '../components/charts/MetricChart'
import FusionWeightChart from '../components/charts/FusionWeightChart'
import useTrainingStore from '../store/trainingStore'

export default function Dashboard() {
  const { status, currentRound, totalRounds, trainingHistory, modelName, modelLoaded, modelVersion, availableModels, switchModel } = useTrainingStore()
  
  const trainingStatus = status;
  const history = trainingHistory || [];

  const progressPct = totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;

  const isInitializing = ['running', 'initializing', 'training'].includes(trainingStatus) && history.length === 0
  const last = history[history.length - 1] || {}
  const fusionData = history.map(h => ({
    round: h.round,
    alpha: h.fusion_weights?.[0] ?? 0.33,
    beta: h.fusion_weights?.[1] ?? 0.33,
    gamma: h.fusion_weights?.[2] ?? 0.34,
  }))

  return (
    <div className="flex flex-col h-full">
      <Header title="Dashboard" />
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Progress bar */}
        {['running', 'initializing', 'training'].includes(trainingStatus) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-slate-900/60 border border-white/5 rounded-xl p-4">
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-2">
              <span>Training Progress</span>
              <span>Round {currentRound} / {totalRounds}</span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-cyan-400 to-emerald-400 rounded-full"
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <p className="text-xs text-slate-500 mt-1">{progressPct.toFixed(1)}% complete</p>
          </motion.div>
        )}

        {/* Initializing State */}
        {isInitializing && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="bg-slate-900/60 border border-cyan-500/20 rounded-xl p-8 text-center space-y-3">
            <div className="flex justify-center">
              <div className="animate-spin h-8 w-8 border-2 border-cyan-400 border-t-transparent rounded-full" />
            </div>
            <div>
              <p className="text-sm text-cyan-400 font-mono font-semibold uppercase tracking-widest animate-pulse">
                Initializing Models / Preparing Round 1...
              </p>
              <p className="text-xs text-slate-500 mt-2 max-w-md mx-auto">
                Loading CLIP & Flan-T5 models, partitioning datasets, and distributing prompts to clients. This typically takes 30-60 seconds.
              </p>
            </div>
          </motion.div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard label="Round" value={currentRound || '—'} sub={`of ${totalRounds}`} color="cyan" icon={Activity} />
          <StatCard label="Loss" value={last.avg_loss?.toFixed(4) ?? '—'} color="rose" icon={TrendingUp} />
          <StatCard label="CIDEr" value={last.cider?.toFixed(3) ?? '—'} color="green" icon={Zap} />
          <StatCard label="BLEU-4" value={last.bleu4?.toFixed(3) ?? '—'} color="amber" icon={Cpu} />
          <StatCard label="Clients" value={last.n_clients ?? '—'} sub="active this round" color="violet" icon={Network} />
          <StatCard label="Comm (MB)" value={last.comm_bytes ? (last.comm_bytes / 1e6).toFixed(2) : '—'} color="cyan" icon={Database} />
        </div>

        <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5 grid gap-4 md:grid-cols-3">
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-[0.3em] mb-2">Active Model</p>
            <p className="text-white font-semibold">{modelName}</p>
            <p className="text-slate-400 text-xs mt-1">Version: {modelVersion}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-[0.3em] mb-2">Loaded</p>
            <p className={modelLoaded ? 'text-emerald-400 font-semibold' : 'text-amber-400 font-semibold'}>
              {modelLoaded ? '✔ Loaded' : '❌ Cold Start'}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-xs text-slate-500 uppercase tracking-[0.3em]">Switch Model</p>
            <select
              value={modelName}
              onChange={(e) => switchModel(e.target.value)}
              className="w-full rounded-lg bg-slate-950 border border-white/10 px-3 py-2 text-sm text-white"
            >
              <option value="default">default</option>
              {availableModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Charts row */}
        {history.length > 0 && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="bg-slate-900/60 border border-white/5 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3">Training Loss</h3>
                <LossChart data={history} />
              </div>
              <div className="bg-slate-900/60 border border-white/5 rounded-xl p-4">
                <h3 className="text-sm font-semibold text-white mb-3">Caption Metrics</h3>
                <MetricChart data={history} />
              </div>
            </div>
            <div className="bg-slate-900/60 border border-white/5 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Adaptive Fusion Weights (α, β, γ) Over Rounds</h3>
              <FusionWeightChart data={fusionData} />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
