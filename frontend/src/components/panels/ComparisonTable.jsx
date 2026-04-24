export default function ComparisonTable({ data }) {
  if (!data?.length) return <p className="text-slate-500 text-sm">No data.</p>
  const cols = ['method','bleu4','rouge1','cider','r@1','clip_sim','comm_cost_mb','conv_round']
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-white/5">
            {cols.map(c => <th key={c} className="py-2 px-3 text-left text-slate-400 uppercase tracking-wider">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className="border-b border-white/5 hover:bg-white/3 transition-colors">
              {cols.map(c => (
                <td key={c} className={`py-2.5 px-3 ${c==='method'?'text-white font-sans text-xs':'text-cyan-300'}`}>
                  {typeof row[c] === 'number' ? row[c].toFixed ? row[c].toFixed(3) : row[c] : row[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
