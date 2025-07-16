// src/components/forms/VectorSetForm.tsx

import { useState } from "react";
import type {
  VectorSetConfig,
  LengthChunkerConfig,
  SentenceChunkerConfig,
  ChunkerConfig,
  EmbedderConfig,
  BGEEmbedderConfig
} from "../../../types/app";
import { EMBEDDER_MODELS, CHUNKERS } from "../../../lib/config";

type Props = {
  dataset: string;
  vectorSets: VectorSetConfig[];
  onChange: (sets: VectorSetConfig[]) => void;
};

const AVAILABLE_CHANNELS = ["abstract", "full_text", "metadata"];

export default function VectorSetForm({ dataset, vectorSets, onChange }: Props) {
  const [channel, setChannel] = useState("");
  const [embedderModel, setEmbedderModel] = useState("");
  const [embedderType, setEmbedderType] = useState<"bge" | "auto_model">("bge");
  const [chunkerType, setChunkerType] = useState<"length_chunker" | "sentence_chunker">("length_chunker");
  const [lengthChunker] = useState<LengthChunkerConfig>({
    type: "length_chunker",
    chunk_size: 512,
    overlap: 50
  });

  const [sentenceChunker] = useState<SentenceChunkerConfig>({
    type: "sentence_chunker",
    language: "en",
  });

  const [bgeEmbedder, setBGEEmbedder] = useState<BGEEmbedderConfig>({
    type: "bge",
    embedding_type: "sparse",
    model_name: EMBEDDER_MODELS.bge[0] // Default to first BGE model
  });

  const [autoModelEmbedder, setAutoModelEmbedder] = useState<EmbedderConfig>({
    type: "auto_model",
    embedding_type: "dense",
    model_name: EMBEDDER_MODELS.auto_model[0] // Default to first auto model
  });

  const handleAdd = () => {
    if (!channel || !embedderModel) return;

    const embedder: EmbedderConfig = embedderType === "bge" ? bgeEmbedder: autoModelEmbedder;
    const chunker: ChunkerConfig = chunkerType === "length_chunker" ? lengthChunker : sentenceChunker;

    const newSet: VectorSetConfig = {
      dataset,
      channel,
      embedder,
      chunker
    };

    onChange([...vectorSets, newSet]);

    // Reset
    setChannel("");
    setEmbedderModel("");
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <label>
        <div>Channel</div>
        <select value={channel} onChange={(e) => setChannel(e.target.value)}>
          <option value="">Select channel</option>
          {AVAILABLE_CHANNELS.map((ch) => (
            <option key={ch} value={ch}>
              {ch}
            </option>
          ))}
        </select>
      </label>

      <label>
        <div>Embedder Type</div>
        <select
          value={embedderType}
          onChange={(e) => {
            const newType = e.target.value as "bge" | "auto_model";
            setEmbedderType(newType);
            setEmbedderModel(""); // Reset model selection when type changes
          }}
        >
          <option value="bge">BGE (Sparse)</option>
          <option value="auto_model">Auto Model (Dense)</option>
        </select>
      </label>


      <label>
        <div>Embedder Model</div>
        <select value={embedderModel} onChange={(e) => {
          setEmbedderModel(e.target.value);
          // Update the appropriate embedder config with the selected model
          if (embedderType === "bge") {
            setBGEEmbedder({
              ...bgeEmbedder,
              model_name: e.target.value
            });
          } else {
            setAutoModelEmbedder({
              ...autoModelEmbedder,
              model_name: e.target.value
            });
          }
        }}>
          <option value="">Select model</option>
          {EMBEDDER_MODELS[embedderType].map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </label>

      <label>
        <div>Chunker Type</div>
        <select
          value={chunkerType}
          onChange={(e) => {
            const newType = e.target.value as "length_chunker" | "sentence_chunker";
            setChunkerType(newType);
          }}
        >
          {CHUNKERS.map((c) => (
            <option key={c.type} value={c.type}>
              {c.type}
            </option>
          ))}
        </select>
      </label>

      <button onClick={handleAdd} disabled={!channel || !embedderModel}>
        Add Vector Set
      </button>

      {vectorSets.length > 0 && (
        <div>
          <h4>Configured Sets</h4>
          <ul>
            {vectorSets.map((vs, i) => (
              <li key={i}>
                {vs.channel} — {vs.embedder.type} / {vs.chunker.type}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
