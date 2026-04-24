import React from "react";
import { AlertTriangle, CheckCircle } from "lucide-react";

export default function DiagnosticsPanel({ diagnostics }) {
  if (!diagnostics) return <div className="text-afspl-muted text-sm font-mono">No diagnostics data.</div>;
  const risk = diagnostics.prompt_collapse_risk || 0;
  const riskColor = risk < 0.3 ? "text-afspl-green" : risk < 0.6 ? "text-afspl-amber" : "text-afspl-red";
  return (
    <div className="space-y-4">
      <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
        <div className="text-xs text-afspl-muted font-mono mb-1">PROMPT COLLAPSE RISK</div>
        <div className={`text-2xl font-display font-bold ${riskColor}`}>{(risk * 100).toFixed(1)}%</div>
      </div>
      <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
        <div className="text-xs text-afspl-muted font-mono mb-2">WARNINGS</div>
        {diagnostics.warnings?.length ? diagnostics.warnings.map((w, i) => (
          <div key={i} className="flex gap-2 text-xs font-mono text-afspl-amber py-1">
            <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
            {w}
          </div>
        )) : (
          <div className="flex gap-2 text-xs font-mono text-afspl-green">
            <CheckCircle size={12} /> No active warnings
          </div>
        )}
      </div>
      <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
        <div className="text-xs text-afspl-muted font-mono mb-2">TOKEN COVERAGE</div>
        <div className="text-xs font-mono text-afspl-text space-y-1">
          {diagnostics.sparse_coverage && Object.entries({
            "Mean Coverage": diagnostics.sparse_coverage.mean_coverage?.toFixed(2),
            "Min Coverage": diagnostics.sparse_coverage.min_coverage,
            "Max Coverage": diagnostics.sparse_coverage.max_coverage,
          }).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-afspl-muted">{k}</span>
              <span>{v ?? "—"}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
