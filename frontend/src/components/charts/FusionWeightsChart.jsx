import React from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";

const COLORS = ["#00d4ff", "#00ff9d", "#ffb800"];
const LABELS = ["α (Global)", "β (Local)", "γ (Modality)"];

export default function FusionWeightsChart({ weights = [0.33, 0.33, 0.34] }) {
  const data = weights.map((v, i) => ({ name: LABELS[i], value: parseFloat((v * 100).toFixed(1)) }));
  return (
    <ResponsiveContainer width="100%" height={180}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value">
          {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
        </Pie>
        <Tooltip
          contentStyle={{ background: "#1a2235", border: "1px solid #1e3a5f", borderRadius: 8, fontFamily: "monospace", fontSize: 11 }}
          formatter={(v) => `${v}%`}
        />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: "monospace" }} />
      </PieChart>
    </ResponsiveContainer>
  );
}
