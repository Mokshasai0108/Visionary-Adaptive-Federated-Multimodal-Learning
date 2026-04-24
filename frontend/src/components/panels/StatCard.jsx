import { motion } from 'framer-motion'
import clsx from 'clsx'

export default function StatCard({ label, value, sub, color = 'cyan', icon: Icon, trend }) {
  const colors = {
    cyan: 'text-cyan-400 bg-cyan-400/10 border-cyan-400/20',
    green: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20',
    amber: 'text-amber-400 bg-amber-400/10 border-amber-400/20',
    rose: 'text-rose-400 bg-rose-400/10 border-rose-400/20',
    violet: 'text-violet-400 bg-violet-400/10 border-violet-400/20',
  }
  return (
    <motion.div whileHover={{ y: -2, scale: 1.01 }} transition={{ type: 'spring', stiffness: 400 }}
      className="bg-slate-900/60 border border-white/5 rounded-xl p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400 font-mono uppercase tracking-wider">{label}</span>
        {Icon && <div className={clsx('w-7 h-7 rounded-md flex items-center justify-center border', colors[color])}><Icon size={13}/></div>}
      </div>
      <div className="flex items-end justify-between">
        <span className={clsx('text-2xl font-semibold font-mono', `text-${color==='cyan'?'cyan':color==='green'?'emerald':color==='amber'?'amber':color==='rose'?'rose':'violet'}-400`)}>{value ?? '—'}</span>
        {trend != null && <span className={clsx('text-xs font-mono', trend >= 0 ? 'text-emerald-400' : 'text-rose-400')}>{trend >= 0 ? '+' : ''}{trend}%</span>}
      </div>
      {sub && <span className="text-xs text-slate-500">{sub}</span>}
    </motion.div>
  )
}
