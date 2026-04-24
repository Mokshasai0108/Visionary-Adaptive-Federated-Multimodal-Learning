import { useEffect, useState } from 'react'
import Header from '../components/layout/Header'
import PromptWeightsPanel from '../components/panels/PromptWeightsPanel'
import { promptsAPI } from '../services/api'
import { motion } from 'framer-motion'

function NormBar({ label, value, max = 3 }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-mono"><span className="text-slate-400">{label}</span><span className="text-white">{value?.toFixed(4)}</span></div>
      <div className="h-1.5 bg-slate-800 rounded-full"><div className="h-full bg-gradient-to-r from-cyan-400 to-violet-400 rounded-full" style={{ width: `${Math.min((value / max) * 100, 100)}%` }} /></div>
    </div>
  )
}

export default function PromptInspector() {
  const [data, setData] = useState(null)
  useEffect(() => { promptsAPI.get().then(r => setData(r.data)).catch(() => {}) }, [])
  return (
    <div className="flex flex-col h-full">
      <Header title="Prompt Inspector" />
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[['Global Prompt Pg', data?.global_prompt_norm, 'cyan'], ['Local Prompt Pl', data?.local_prompt_norm, 'emerald'], ['Modality Prompt Pm', data?.modality_prompt_norm, 'amber']].map(([label, norm, _color]) => (
            <div key={label} className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
              <p className="text-xs font-mono text-slate-400 mb-3 uppercase">{label}</p>
              <p className="text-2xl font-mono text-white mb-1">{norm?.toFixed(4) ?? '—'}</p>
              <p className="text-xs text-slate-500">L2 norm</p>
            </div>
          ))}
        </div>
        <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
          <p className="text-sm font-semibold text-white mb-4">Adaptive Fusion Weights (α, β, γ)</p>
          <PromptWeightsPanel weights={data?.fusion_weights_avg} />
        </div>
        <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
          <p className="text-sm font-semibold text-white mb-4">Token Coverage (per prompt token, all rounds)</p>
          <div className="flex gap-1 flex-wrap">
            {(data?.token_coverage || Array(16).fill(0)).map((v, i) => (
              <motion.div key={i} whileHover={{ scale: 1.2 }} title={`Token ${i}: ${v} updates`}
                className="w-7 h-7 rounded text-xs font-mono flex items-center justify-center"
                style={{ background: `rgba(0,229,255,${Math.min(v / 30, 1) * 0.6 + 0.1})`, color: '#e2e8f0' }}>
                {v}
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
