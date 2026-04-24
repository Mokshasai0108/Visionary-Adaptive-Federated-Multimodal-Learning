import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export default function FusionWeightChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
        <defs>
          <linearGradient id="alpha" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3}/><stop offset="95%" stopColor="#00e5ff" stopOpacity={0}/></linearGradient>
          <linearGradient id="beta" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#00ffa3" stopOpacity={0.3}/><stop offset="95%" stopColor="#00ffa3" stopOpacity={0}/></linearGradient>
          <linearGradient id="gamma" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ffb800" stopOpacity={0.3}/><stop offset="95%" stopColor="#ffb800" stopOpacity={0}/></linearGradient>
        </defs>
        <XAxis dataKey="round" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0,1]} />
        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
        <Area type="monotone" dataKey="alpha" stroke="#00e5ff" fill="url(#alpha)" strokeWidth={1.5} name="α (Global)" />
        <Area type="monotone" dataKey="beta" stroke="#00ffa3" fill="url(#beta)" strokeWidth={1.5} name="β (Local)" />
        <Area type="monotone" dataKey="gamma" stroke="#ffb800" fill="url(#gamma)" strokeWidth={1.5} name="γ (Modality)" />
      </AreaChart>
    </ResponsiveContainer>
  )
}
