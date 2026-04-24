import { useEffect, useState } from 'react'
import Header from '../components/layout/Header'
import ComparisonTable from '../components/panels/ComparisonTable'
import { metricsAPI } from '../services/api'

export default function Results() {
  const [data, setData] = useState(null)
  useEffect(() => { metricsAPI.get().then(r => setData(r.data)).catch(() => {}) }, [])
  return (
    <div className="flex flex-col h-full">
      <Header title="Results" />
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white mb-4">Experiment Comparison Table</h3>
          <ComparisonTable data={data?.comparison_table} />
        </div>
        {data?.summary && Object.keys(data.summary).length > 0 && (
          <div className="bg-slate-900/60 border border-white/5 rounded-xl p-5">
            <h3 className="text-sm font-semibold text-white mb-3">Training Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(data.summary).map(([k, v]) => (
                <div key={k} className="text-center">
                  <p className="text-xs font-mono text-slate-400 uppercase">{k.replace(/_/g, ' ')}</p>
                  <p className="text-lg font-mono text-cyan-400 mt-1">{typeof v === 'number' ? v.toFixed(3) : v}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
