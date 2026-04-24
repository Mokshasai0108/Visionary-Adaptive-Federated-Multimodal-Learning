import React from "react";

export default function StatCard({ label, value, sub, accent = false, icon: Icon }) {
  return (
    <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-afspl-muted font-mono uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={14} className="text-afspl-muted" />}
      </div>
      <div className={`text-2xl font-display font-bold ${accent ? "text-afspl-accent" : "text-afspl-text"}`}>
        {value ?? "—"}
      </div>
      {sub && <div className="text-xs text-afspl-muted mt-1 font-mono">{sub}</div>}
    </div>
  );
}
