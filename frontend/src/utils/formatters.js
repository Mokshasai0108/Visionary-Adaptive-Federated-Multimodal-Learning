export const fmtBytes = (bytes) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

export const fmtPct = (v) => `${(v * 100).toFixed(1)}%`;
export const fmtNum = (v, d = 4) => typeof v === "number" ? v.toFixed(d) : "—";
export const fmtRound = (cur, total) => `${cur} / ${total}`;
