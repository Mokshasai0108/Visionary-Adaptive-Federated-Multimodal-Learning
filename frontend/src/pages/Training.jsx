import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Square, Settings } from 'lucide-react'
import Header from '../components/layout/Header'
import { trainAPI } from '../services/api'
import { useStore } from '../store/useStore'

const Field = ({ label, children }) => (
  <div>
    <label className="block text-xs font-mono text-slate-400 mb-1.5 uppercase tracking-wider">{label}</label>
    {children}
  </div>
)
const Input = (props) => <input {...props} className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-cyan-400/50 transition-colors" />
const Select = ({ children, ...props }) => <select {...props} className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-cyan-400/50">{children}</select>

export default function Training() {
  const { trainingStatus, trainingConfig, setTrainingConfig } = useStore()
  const [msg, setMsg] = useState('')

  const set = (k, v) => setTrainingConfig(k, v)

  const start = async () => {
    try {
      await trainAPI.start(trainingConfig)
      setMsg('Training started!')
    } catch (e) {
      setMsg(e.response?.data?.detail || 'Error starting training')
    }
  }
  const stop = async () => {
    try {
      await trainAPI.stop()
      setMsg('Stop signal sent.')
    } catch (e) {
      setMsg(e.response?.data?.detail || 'Error stopping training')
    }
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Training Control" />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl">
          <div className="bg-slate-900/60 border border-white/5 rounded-xl p-6 space-y-5">
            <div className="flex items-center gap-2 mb-2">
              <Settings size={16} className="text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">Federated Training Configuration</h2>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Rounds"><Input type="number" value={trainingConfig.num_rounds} onChange={e => set('num_rounds', +e.target.value)} min={1} max={100} /></Field>
              <Field label="Clients"><Input type="number" value={trainingConfig.num_clients} onChange={e => set('num_clients', +e.target.value)} min={1} max={50} /></Field>
              <Field label="Prompt Length"><Input type="number" value={trainingConfig.prompt_length} onChange={e => set('prompt_length', +e.target.value)} min={10} max={20} /></Field>
              <Field label="Fusion Strategy">
                <Select value={trainingConfig.fusion_strategy} onChange={e => set('fusion_strategy', e.target.value)}>
                  <option value="dynamic">Dynamic Gating</option>
                  <option value="learnable">Learnable Scalar</option>
                  <option value="static">Static Fixed</option>
                </Select>
              </Field>
              <Field label="Top-K Ratio"><Input type="number" value={trainingConfig.k_ratio} onChange={e => set('k_ratio', +e.target.value)} min={0.1} max={1} step={0.05} /></Field>
              <Field label="Seed"><Input type="number" value={trainingConfig.seed} onChange={e => set('seed', +e.target.value)} /></Field>
              <Field label="λ1 (CE weight)"><Input type="number" value={trainingConfig.lambda1} onChange={e => set('lambda1', +e.target.value)} step={0.1} /></Field>
              <Field label="λ2 (CLIP weight)"><Input type="number" value={trainingConfig.lambda2} onChange={e => set('lambda2', +e.target.value)} step={0.1} /></Field>
            </div>
            <div className="flex gap-3 pt-2">
              <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={start}
                disabled={trainingStatus === 'running'}
                className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-emerald-500 text-slate-900 font-semibold text-sm rounded-lg disabled:opacity-40 disabled:cursor-not-allowed">
                <Play size={14} />Start Training
              </motion.button>
              {trainingStatus === 'running' && (
                <motion.button whileHover={{ scale: 1.02 }} onClick={stop}
                  className="flex items-center gap-2 px-5 py-2.5 bg-rose-500/20 border border-rose-400/30 text-rose-400 font-semibold text-sm rounded-lg">
                  <Square size={14} />Stop
                </motion.button>
              )}
            </div>
            {msg && <p className="text-xs font-mono text-emerald-400 mt-2">{msg}</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
