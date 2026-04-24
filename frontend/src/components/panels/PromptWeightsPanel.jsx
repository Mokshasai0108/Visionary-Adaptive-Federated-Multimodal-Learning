export default function PromptWeightsPanel({ weights }) {
  const items = [
    { label: 'α — Global Prompt', value: weights?.[0] ?? 0.35, color: 'bg-cyan-400' },
    { label: 'β — Local Prompt',  value: weights?.[1] ?? 0.38, color: 'bg-emerald-400' },
    { label: 'γ — Modality Prompt', value: weights?.[2] ?? 0.27, color: 'bg-amber-400' },
  ]
  return (
    <div className="space-y-3">
      {items.map(({ label, value, color }) => (
        <div key={label}>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400 font-mono">{label}</span>
            <span className="text-white font-mono">{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${value*100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}
