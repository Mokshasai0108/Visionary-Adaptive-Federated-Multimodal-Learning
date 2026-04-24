import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'

export default function MetricChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey="round" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(0,229,255,0.1)', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
        <Line type="monotone" dataKey="cider" stroke="#00ffa3" strokeWidth={2} dot={false} name="CIDEr" />
        <Line type="monotone" dataKey="bleu4" stroke="#ffb800" strokeWidth={2} dot={false} name="BLEU-4" />
      </LineChart>
    </ResponsiveContainer>
  )
}
