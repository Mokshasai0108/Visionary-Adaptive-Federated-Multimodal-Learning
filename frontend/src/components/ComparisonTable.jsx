import React from "react";

export default function ComparisonTable({ data = [] }) {
  if (!data.length) return <div className="text-afspl-muted text-sm font-mono py-8 text-center">No comparison data yet. Complete training to generate results.</div>;
  const cols = Object.keys(data[0]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-afspl-border">
            {cols.map(c => <th key={c} className="text-left py-2 px-3 text-afspl-muted uppercase tracking-wider whitespace-nowrap">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} className={`border-b border-afspl-border/50 hover:bg-afspl-card transition-colors ${row.Method?.includes("AFSPL Full") ? "text-afspl-accent" : "text-afspl-text"}`}>
              {cols.map(c => <td key={c} className="py-2 px-3 whitespace-nowrap">{String(row[c] ?? "—")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
