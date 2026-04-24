import React, { useState } from "react";
import useTrainingStore from "../store/trainingStore";
import { Play, Square } from "lucide-react";

const Field = ({ label, name, type = "number", value, onChange, min, max, step, options }) => (
  <div>
    <label className="block text-xs text-afspl-muted font-mono mb-1 uppercase tracking-wider">{label}</label>
    {options ? (
      <select name={name} value={value} onChange={onChange}
        className="w-full bg-afspl-surface border border-afspl-border rounded-lg px-3 py-2 text-afspl-text text-sm font-mono focus:border-afspl-accent outline-none">
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    ) : (
      <input type={type} name={name} value={value} onChange={onChange} min={min} max={max} step={step}
        className="w-full bg-afspl-surface border border-afspl-border rounded-lg px-3 py-2 text-afspl-text text-sm font-mono focus:border-afspl-accent outline-none" />
    )}
  </div>
);

export default function TrainingControl() {
  const { startTraining, stopTraining, isTraining, error, clearError } = useTrainingStore();
  const [form, setForm] = useState({
    num_rounds: 30, num_clients: 10, fusion_strategy: "dynamic",
    prompt_length: 16, batch_size: 32, learning_rate: 0.0001,
    lambda1: 1.0, lambda2: 0.5, k_ratio: 0.3, seed: 42,
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(f => ({ ...f, [name]: isNaN(value) || value === "" ? value : Number(value) }));
  };

  const handleStart = () => { clearError(); startTraining(form); };

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-display font-bold text-afspl-text">Training Control</h1>
        <p className="text-afspl-muted text-sm font-mono mt-1">Configure and launch federated training</p>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-afspl-red rounded-xl p-4 text-afspl-red text-sm font-mono">
          {error}
        </div>
      )}

      <div className="bg-afspl-card border border-afspl-border rounded-xl p-6 space-y-4">
        <h3 className="text-sm font-display font-semibold text-afspl-text">Federated Settings</h3>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Rounds" name="num_rounds" value={form.num_rounds} onChange={handleChange} min={1} max={100} />
          <Field label="Clients" name="num_clients" value={form.num_clients} onChange={handleChange} min={2} max={50} />
          <Field label="Fusion Strategy" name="fusion_strategy" value={form.fusion_strategy} onChange={handleChange} options={["static","learnable","dynamic"]} />
          <Field label="Prompt Length" name="prompt_length" value={form.prompt_length} onChange={handleChange} min={10} max={20} />
        </div>
      </div>

      <div className="bg-afspl-card border border-afspl-border rounded-xl p-6 space-y-4">
        <h3 className="text-sm font-display font-semibold text-afspl-text">Optimization</h3>
        <div className="grid grid-cols-2 gap-4">
          <Field label="Batch Size" name="batch_size" value={form.batch_size} onChange={handleChange} min={4} max={128} />
          <Field label="Learning Rate" name="learning_rate" value={form.learning_rate} onChange={handleChange} step={0.00001} />
          <Field label="λ1 (CE Weight)" name="lambda1" value={form.lambda1} onChange={handleChange} step={0.1} />
          <Field label="λ2 (CLIP Weight)" name="lambda2" value={form.lambda2} onChange={handleChange} step={0.1} />
          <Field label="Top-K Ratio" name="k_ratio" value={form.k_ratio} onChange={handleChange} step={0.05} min={0.1} max={1.0} />
          <Field label="Seed" name="seed" value={form.seed} onChange={handleChange} />
        </div>
      </div>

      <div className="flex gap-3">
        <button onClick={handleStart} disabled={isTraining}
          className="flex items-center gap-2 px-6 py-3 bg-afspl-accent text-afspl-bg font-display font-semibold rounded-xl hover:opacity-90 disabled:opacity-40 transition-all">
          <Play size={16} /> Start Training
        </button>
        {isTraining && (
          <button onClick={stopTraining}
            className="flex items-center gap-2 px-6 py-3 bg-afspl-red text-white font-display font-semibold rounded-xl hover:opacity-90 transition-all">
            <Square size={16} /> Stop
          </button>
        )}
      </div>
    </div>
  );
}
