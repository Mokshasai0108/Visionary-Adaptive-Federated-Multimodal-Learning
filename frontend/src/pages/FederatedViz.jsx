import React from "react";
import useTrainingStore from "../store/trainingStore";
import { fmtBytes } from "../utils/formatters";

function ClientNode({ id, active, loss }) {
  return (
    <div className={`flex flex-col items-center gap-1 ${active ? "" : "opacity-40"}`}>
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-xs font-mono font-bold border-2 transition-all
        ${active ? "border-afspl-accent text-afspl-accent bg-afspl-accent/10" : "border-afspl-border text-afspl-muted"}`}>
        C{id}
      </div>
      {loss != null && <span className="text-xs font-mono text-afspl-muted">{loss.toFixed(3)}</span>}
    </div>
  );
}

export default function FederatedViz() {
  const { trainingHistory, currentRound, totalCommBytes } = useTrainingStore();
  const latest = trainingHistory[trainingHistory.length - 1] || {};
  const activeClients = Math.floor((latest.round || 0) % 10) + 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-display font-bold text-afspl-text">Federation Visualization</h1>
        <p className="text-afspl-muted text-sm font-mono mt-1">Client topology, global prompt flow, communication stats</p>
      </div>

      {/* Server node */}
      <div className="bg-afspl-card border border-afspl-border rounded-xl p-6">
        <div className="flex flex-col items-center gap-6">
          <div className="w-20 h-20 rounded-2xl bg-afspl-accent/10 border-2 border-afspl-accent flex flex-col items-center justify-center">
            <span className="text-afspl-accent font-mono font-bold text-sm">SERVER</span>
            <span className="text-xs font-mono text-afspl-muted">Pg</span>
          </div>
          <div className="flex items-center gap-2 text-afspl-muted">
            <div className="w-px h-8 bg-afspl-border" />
            <span className="text-xs font-mono">FedAvg + Sparse Reconstruction</span>
            <div className="w-px h-8 bg-afspl-border" />
          </div>
          <div className="grid grid-cols-5 lg:grid-cols-10 gap-4 w-full justify-items-center">
            {Array.from({ length: 10 }, (_, i) => (
              <ClientNode key={i} id={i + 1} active={i < activeClients} loss={latest.train_loss} />
            ))}
          </div>
        </div>
      </div>

      {/* Comm stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
          <div className="text-xs text-afspl-muted font-mono mb-1">TOTAL BYTES SENT</div>
          <div className="text-xl font-display font-bold text-afspl-accent">{fmtBytes(totalCommBytes)}</div>
        </div>
        <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
          <div className="text-xs text-afspl-muted font-mono mb-1">ACTIVE ROUND</div>
          <div className="text-xl font-display font-bold text-afspl-text">{currentRound}</div>
        </div>
        <div className="bg-afspl-card border border-afspl-border rounded-xl p-4">
          <div className="text-xs text-afspl-muted font-mono mb-1">ACTIVE CLIENTS</div>
          <div className="text-xl font-display font-bold text-afspl-green">{activeClients} / 10</div>
        </div>
      </div>
    </div>
  );
}
