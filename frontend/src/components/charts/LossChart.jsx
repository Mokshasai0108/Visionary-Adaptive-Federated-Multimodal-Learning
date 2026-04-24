import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export default function LossChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis dataKey="round" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} label={{ value: 'Round', position: 'insideBottom', fill: '#475569', fontSize: 11 }} />
        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(0,229,255,0.1)', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }} />
        <Line type="monotone" dataKey="avg_loss" stroke="#00e5ff" strokeWidth={2} dot={false} name="Loss" />
      </LineChart>
    </ResponsiveContainer>
  )
}
