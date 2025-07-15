// src/components/forms/RerankerForm.tsx

import type { RerankerConfig } from "../../../types/app";

type Props = {
  reranker: RerankerConfig;
  onChange: (reranker: RerankerConfig) => void;
};

export default function RerankerForm({ reranker, onChange }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <label>
        <div>Reranker Type</div>
        <select
          value={reranker.type}
          onChange={(e) => onChange({ type: e.target.value as RerankerConfig["type"] })}
        >
          <option value="identity">identity</option>
        </select>
      </label>
    </div>
  );
}
