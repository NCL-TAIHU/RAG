// src/components/forms/ReviewForm.tsx

import type { AppConfig } from "../../../types/app";
import { createApp } from "../../../lib/api";

type Props = {
  metadata: AppConfig;
  onSubmit: () => void;
};

export default function ReviewForm({ metadata, onSubmit }: Props) {
    const handleSubmit = async () => {
        try {
            await createApp(metadata);
            alert("App created!");
            onSubmit(); // maybe navigate or reset state
        } catch (err) {
            alert("Failed to create app: " + (err as Error).message);
        }
    };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <h3>Review App Configuration</h3>

      <div>
        <strong>Name:</strong> {metadata.name}
      </div>

      <div>
        <strong>Dataset:</strong> {metadata.dataset}
      </div>

      <div>
        <strong>Router:</strong> {metadata.router.type}
      </div>

      <div>
        <strong>Reranker:</strong> {metadata.reranker?.type ?? "None"}
      </div>

      <div>
        <strong>Search Engines:</strong>
        <ul>
          {metadata.search_engines.map((engine, idx) => (
            <li key={idx}>
              <code>{engine.type}</code>
            </li>
          ))}
        </ul>
      </div>

      <button onClick={handleSubmit}>Create App</button>
    </div>
  );
}
