import React, { useState } from "react";
import axios from "axios";
import { Upload, Sparkles, Cpu, Zap, Settings2 } from "lucide-react";
import useTrainingStore from "../store/trainingStore";

export default function Inference() {
  const { lastFusionWeights } = useTrainingStore();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prefix, setPrefix] = useState("");
  const [mode, setMode] = useState("guided"); // guided | full
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const handleInfer = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("text_prompt", prefix);
      fd.append("mode", mode);

      // Using direct axios call for multimodal multipart support
      const res = await axios.post("http://localhost:8000/infer", fd, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      setResult({ 
        caption: "Inference failed. Ensure model is ready and server is running.", 
        clip_similarity: 0,
        confidence: 0,
        attribution: { prefix: 0, prompts: 0, vision: 0 },
        ablation: { use_prefix: false, use_prompts: false, use_image: false }
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-3xl font-display font-bold text-afspl-text flex items-center gap-3">
          <Zap className="text-afspl-accent" /> Inference Workspace
        </h1>
        <p className="text-afspl-muted text-sm font-mono mt-1">
          Phase 2.6 Hardened: Multimodal Steering + XAI Attribution + Confidence Metrics
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="bg-afspl-card border border-afspl-border rounded-xl p-6 space-y-4">
            <label className="cursor-pointer flex flex-col items-center justify-center border-2 border-dashed border-afspl-border rounded-xl aspect-[4/3] hover:border-afspl-accent transition-colors overflow-hidden">
              {preview ? (
                <img src={preview} alt="preview" className="w-full h-full object-cover" />
              ) : (
                <><Upload size={48} className="text-afspl-muted mb-2" /><span className="text-afspl-muted text-sm font-mono text-center px-4">Upload research asset</span></>
              )}
              <input type="file" accept="image/*" onChange={handleFile} className="hidden" />
            </label>
          </div>

          <div className="bg-afspl-card border border-afspl-border rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-mono text-afspl-muted uppercase tracking-widest flex items-center gap-2">
              <Settings2 size={14} /> Steering Control
            </h3>
            
            <div className="space-y-2">
              <label className="text-[10px] text-afspl-muted font-mono">CONTEXT HINT (PREFIX)</label>
              <textarea
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="A high-quality photo of..."
                className="w-full bg-afspl-bg border border-afspl-border rounded-lg p-3 text-afspl-text text-sm font-mono focus:border-afspl-accent outline-none min-h-[80px]"
              />
            </div>

            <button
              onClick={handleInfer}
              disabled={!file || loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-afspl-accent text-afspl-bg font-display font-semibold rounded-xl hover:opacity-90 disabled:opacity-40 transition-all shadow-lg shadow-afspl-accent/20"
            >
              <Sparkles size={16} />
              {loading ? "Synthesizing Core..." : "Run Research Pass"}
            </button>
          </div>
        </div>

        <div className="space-y-6">
          {result ? (
            <div className="bg-afspl-card border border-afspl-border rounded-2xl p-6 shadow-2xl space-y-6 animate-in fade-in slide-in-from-right-4">
              <div>
                <div className="text-[10px] text-afspl-muted font-mono uppercase tracking-[0.2em] mb-3 flex items-center gap-2">
                  <Sparkles size={12} className="text-afspl-accent" /> Grounded Output
                </div>
                <div className="p-4 bg-afspl-bg/50 border border-afspl-border border-l-4 border-l-afspl-accent rounded-lg">
                  <p className="text-afspl-text font-display text-xl leading-relaxed italic">
                    "{result.caption}"
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-afspl-bg rounded-xl p-4 border border-afspl-border">
                  <div className="text-[10px] text-afspl-muted font-mono mb-1">CONFIDENCE (GM)</div>
                  <div className="flex items-end gap-2">
                    <div className="text-2xl font-display font-bold text-afspl-accent">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="h-4 w-full bg-afspl-card rounded-full overflow-hidden mb-2">
                        <div className="h-full bg-afspl-accent" style={{width: `${result.confidence*100}%`}} />
                    </div>
                  </div>
                </div>
                
                <div className="bg-afspl-bg rounded-xl p-4 border border-afspl-border">
                  <div className="text-[10px] text-afspl-muted font-mono mb-1">SEMANTIC ALIGNMENT</div>
                  <div className="text-2xl font-display font-bold text-emerald-500">
                    {(result.clip_similarity * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-[10px] text-afspl-muted font-mono uppercase tracking-wider">Attribution Heatmap (XAI)</div>
                <div className="h-8 w-full flex rounded-lg overflow-hidden border border-afspl-border">
                    <div className="h-full bg-blue-500 transition-all flex items-center justify-center text-[10px] font-bold" style={{width: `${result.attribution.prefix*100}%`}}>
                        {result.attribution.prefix > 0.1 && "PREFIX"}
                    </div>
                    <div className="h-full bg-purple-500 transition-all flex items-center justify-center text-[10px] font-bold" style={{width: `${result.attribution.prompts*100}%`}}>
                        {result.attribution.prompts > 0.1 && "PROMPTS"}
                    </div>
                    <div className="h-full bg-emerald-500 transition-all flex items-center justify-center text-[10px] font-bold" style={{width: `${result.attribution.vision*100}%`}}>
                        {result.attribution.vision > 0.1 && "VISION"}
                    </div>
                </div>
                <div className="flex justify-between text-[8px] font-mono text-afspl-muted px-1">
                    <span>{ (result.attribution.prefix*100).toFixed(0) }%</span>
                    <span>{ (result.attribution.prompts*100).toFixed(0) }%</span>
                    <span>{ (result.attribution.vision*100).toFixed(0) }%</span>
                </div>
              </div>

              <div className="pt-4 border-t border-afspl-border">
                  <div className="flex items-center justify-between">
                     <div className="text-[10px] text-afspl-muted font-mono">ABLATION CONFIG</div>
                     <div className="flex gap-2">
                        {Object.entries(result.ablation).map(([k, v]) => (
                            <span key={k} className={`px-2 py-0.5 rounded text-[8px] font-mono border ${v ? 'border-emerald-500/50 text-emerald-500' : 'border-red-500/50 text-red-500'}`}>
                                {k.replace('use_', '').toUpperCase()}
                            </span>
                        ))}
                     </div>
                  </div>
              </div>

            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center p-12 bg-afspl-card/30 border border-afspl-border border-dashed rounded-2xl text-afspl-muted">
              <Sparkles size={48} className="opacity-20 mb-4" />
              <p className="font-mono text-sm">Upload an image to see research diagnostics</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
