import React, { useState } from "react";
import axios from "axios";
import { Sparkles, Replace, Activity, Layers, Sliders, Info, Zap } from "lucide-react";
import useTrainingStore from "../store/trainingStore";

export default function Mirror() {
  const { lastFusionWeights } = useTrainingStore();
  const [prompt, setPrompt] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("compare"); // "compare" | "single"

  const handleDream = async () => {
    if (!prompt) return;
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/api/mirror/dream", {
        prompt,
        alpha: parseFloat(alpha),
        seed: 42
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div>
        <h1 className="text-3xl font-display font-bold text-afspl-text flex items-center gap-3">
          <Replace className="text-afspl-accent" /> Mirror Workspace
        </h1>
        <p className="text-afspl-muted text-sm font-mono mt-1">
          Phase 3 Bidirectional Logic: Steering Generative Distribution with Federated Prompts
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-afspl-card border border-afspl-border rounded-xl p-6 space-y-6">
            <h3 className="text-xs font-mono text-afspl-muted uppercase tracking-widest flex items-center gap-2">
              <Sliders size={14} /> Steering Control
            </h3>

            <div className="space-y-3">
              <label className="text-[10px] text-afspl-muted font-mono uppercase">Target Concept Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A photo of a street in a city..."
                className="w-full bg-afspl-bg border border-afspl-border rounded-lg p-3 text-afspl-text text-sm font-mono focus:border-afspl-accent outline-none min-h-[100px]"
              />
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <label className="text-[10px] text-afspl-muted font-mono uppercase">Steering Strength (α)</label>
                <span className="text-xs font-mono text-afspl-accent">{(alpha * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={alpha}
                onChange={(e) => setAlpha(e.target.value)}
                className="w-full accent-afspl-accent"
              />
              <p className="text-[9px] text-afspl-muted font-mono italic">
                Capped at 50% max influence to preserve prior stability.
              </p>
            </div>

            <button
              onClick={handleDream}
              disabled={!prompt || loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-4 bg-afspl-accent text-afspl-bg font-display font-bold rounded-xl hover:opacity-90 disabled:opacity-40 transition-all shadow-xl shadow-afspl-accent/20"
            >
              {loading ? <Zap className="animate-pulse" size={18} /> : <Sparkles size={18} />}
              {loading ? "Swapping VRAM & Dreaming..." : "Materialize Mirror"}
            </button>
          </div>

          <div className="bg-afspl-card border border-afspl-border rounded-xl p-5 space-y-4">
            <div className="text-[10px] text-afspl-muted font-mono uppercase flex items-center gap-2">
                <Layers size={12} /> Active Research Weights
            </div>
            <div className="space-y-3">
                {[
                  { label: 'GLOBAL', val: lastFusionWeights?.[0] || 0.33, color: 'bg-cyan-500' },
                  { label: 'LOCAL', val: lastFusionWeights?.[1] || 0.33, color: 'bg-emerald-500' },
                  { label: 'MODAL', val: lastFusionWeights?.[2] || 0.34, color: 'bg-purple-500' }
                ].map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[8px] font-mono text-afspl-muted">
                      <span>{item.label}</span>
                      <span>{(item.val * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 w-full bg-afspl-bg rounded-full overflow-hidden border border-afspl-border">
                      <div className={`h-full ${item.color}`} style={{ width: `${item.val * 100}%` }} />
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {result ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-afspl-card border border-afspl-border rounded-2xl overflow-hidden group">
                  <div className="p-3 bg-afspl-bg border-b border-afspl-border flex justify-between items-center">
                    <span className="text-[10px] font-mono font-bold text-afspl-muted">BASELINE SD-TURBO</span>
                    <span className="text-[10px] font-mono text-emerald-500/80">CLIP: {result.metrics.clip_base}</span>
                  </div>
                  <img src={`data:image/png;base64,${result.baseline}`} className="w-full aspect-square object-cover" />
                </div>

                <div className="bg-afspl-card border border-afspl-accent/30 rounded-2xl overflow-hidden relative shadow-2xl shadow-afspl-accent/10">
                  <div className="p-3 bg-afspl-bg border-b border-afspl-accent/30 flex justify-between items-center">
                    <span className="text-[10px] font-mono font-bold text-afspl-accent flex items-center gap-1">
                      <Replace size={12} /> AFSPL-STEERED
                    </span>
                    <span className="text-[10px] font-mono text-afspl-accent">CLIP: {result.metrics.clip_steer}</span>
                  </div>
                  <img src={`data:image/png;base64,${result.steered}`} className="w-full aspect-square object-cover" />
                  <div className="absolute bottom-4 right-4 bg-afspl-bg/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-afspl-accent/20 text-[10px] font-bold text-afspl-accent flex items-center gap-2">
                    <Activity size={10} /> Δ SHIFT: {result.metrics.delta}
                  </div>
                </div>
              </div>

              <div className="bg-afspl-card border border-afspl-border rounded-xl p-6">
                <div className="flex items-center gap-2 mb-4">
                    <Info size={14} className="text-afspl-accent" />
                    <span className="text-[10px] font-mono uppercase tracking-widest text-afspl-muted">Provable Contribution Metric</span>
                </div>
                <div className="h-4 w-full bg-afspl-bg rounded-full overflow-hidden border border-afspl-border p-0.5">
                    <div className="h-full bg-afspl-accent rounded-full transition-all duration-1000" style={{width: `${Math.min(result.metrics.delta * 500, 100)}%`}} />
                </div>
                <p className="text-[10px] text-afspl-muted font-mono mt-3 leading-relaxed">
                   The <span className="text-afspl-accent">Delta Shift</span> measures the semantic distance between pure Stable Diffusion and your Federated distribution. A shift of {result.metrics.delta} indicates meaningful bidirectional knowledge transfer from the learned prompts.
                </p>
              </div>
            </div>
          ) : (
            <div className="h-[500px] flex flex-col items-center justify-center bg-afspl-card/20 border border-afspl-border border-dashed rounded-2xl">
                <Replace size={64} className="opacity-10 mb-6" />
                <p className="font-mono text-afspl-muted text-sm">Define a concept to begin bidirectional materialization</p>
                <div className="mt-4 flex gap-4 text-[10px] font-mono text-afspl-muted/50">
                    <span className="flex items-center gap-1"><Info size={10} /> VRAM DETERMINISTIC</span>
                    <span className="flex items-center gap-1"><Info size={10} /> ADDITIVE STEER</span>
                </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
