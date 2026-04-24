import { useEffect, useState } from 'react'
import Header from '../components/layout/Header'
import { diagnosticsAPI } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { AlertTriangle, CheckCircle } from 'lucide-react'
import clsx from 'clsx'

export default function Diagnostics() {
  const [data, setData] = useState(null)
  useEffect(() => { diagnosticsAPI.get().then(r => setData(r.data)).catch(() => {}) }, [])
  const driftData = data?.client_drift_scores?.map((v, i) => ({ client: `C${i+1}`, drift: v })) || []
  return (
    <div className="flex flex-col h-full">
      <Header title="Diagnostics" />
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            ['Prompt Collapse Risk', data?.prompt_collapse_risk, 0.2, '%'],
            ['Token Coverage', data?.token_coverage_pct, 0.5, '%'],
            ['Comm Anomalies', data?.communication_anomalies?.length, 0, ''],
          ].map(([label, value, threshold, unit]) => (
            <div key={label} className="bg-slate-900/60 border border-white/5 rounded-xl p-4">
              <p className="text-xs font-mono text-slate-400 uppercase mb-2">{label}</p>
              <div className="flex items-center gap-2">
                <p className="text-xl font-mono text-white">{value != null ? (typeof value === 'number' ? (value * (unit === '%' && value <= 1 ? 100 : 1)).toFixed(1) + unit : value) : '—'}</p>
                {typeof value === 'number' && (value > threshold ? <AlertTriangle size={14} className="text-rose-400" /> : <CheckCircle size={14} className="text-emerald-400" />)}
              </div>
            </div>
          ))}
        </div>
        <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Client Drift Scores</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={driftData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="client" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(0,229,255,0.1)', borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="drift" fill="#ff4d6d" radius={[3, 3, 0, 0]} opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
