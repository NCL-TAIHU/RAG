// src/components/forms/VectorSetForm.tsx

import { useState } from "react";
import type {
  VectorManagerConfig,
  SimpleChunkerConfig,
  SentenceChunkerConfig,
  ChunkerConfig,
  EmbedderConfig
} from "../../../types/app";
import { EMBEDDER_MODELS, CHUNKERS } from "../../../lib/config";

type Props = {
  dataset: string;
  vectorSets: VectorManagerConfig[];
  onChange: (sets: VectorManagerConfig[]) => void;
};

const AVAILABLE_CHANNELS = ["abstract", "full_text", "metadata"];

export default function VectorSetForm({ dataset, vectorSets, onChange }: Props) {
  const [channel, setChannel] = useState("");
  const [embedderType, setEmbedderType] = useState<"bge" | "auto_model">("bge");
  const [embedderModel, setEmbedderModel] = useState("");
  const [chunkerType, setChunkerType] = useState<"simple_chunker" | "sentence_chunker">("simple_chunker");

  const [simpleChunker, setSimpleChunker] = useState<SimpleChunkerConfig>({
    type: "simple_chunker",
    chunk_size: 512,
    overlap: 50
  });

  const [sentenceChunker, setSentenceChunker] = useState<SentenceChunkerConfig>({
    type: "sentence_chunker",
    max_length: 512,
    stride: 50
  });

  const handleAdd = () => {
    if (!channel || !embedderModel) return;

    const embedder: EmbedderConfig = {
      type: embedderType,
      model_name: embedderModel
    };

    const chunker: ChunkerConfig =
      chunkerType === "simple_chunker" ? simpleChunker : sentenceChunker;

    const newSet: VectorManagerConfig = {
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
            const t = e.target.value as "bge" | "auto_model";
            setEmbedderType(t);
            setEmbedderModel(""); // reset model on switch
          }}
        >
          {Object.keys(EMBEDDER_MODELS).map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
      </label>

      <label>
        <div>Embedder Model</div>
        <select value={embedderModel} onChange={(e) => setEmbedderModel(e.target.value)}>
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
            const newType = e.target.value as "simple_chunker" | "sentence_chunker";
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

      {chunkerType === "simple_chunker" && (
        <>
          <label>
            Chunk Size
            <input
              type="number"
              value={simpleChunker.chunk_size}
              onChange={(e) =>
                setSimpleChunker({ ...simpleChunker, chunk_size: parseInt(e.target.value) })
              }
            />
          </label>
          <label>
            Overlap
            <input
              type="number"
              value={simpleChunker.overlap}
              onChange={(e) =>
                setSimpleChunker({ ...simpleChunker, overlap: parseInt(e.target.value) })
              }
            />
          </label>
        </>
      )}

      {chunkerType === "sentence_chunker" && (
        <>
          <label>
            Max Length
            <input
              type="number"
              value={sentenceChunker.max_length}
              onChange={(e) =>
                setSentenceChunker({ ...sentenceChunker, max_length: parseInt(e.target.value) })
              }
            />
          </label>
          <label>
            Stride
            <input
              type="number"
              value={sentenceChunker.stride}
              onChange={(e) =>
                setSentenceChunker({ ...sentenceChunker, stride: parseInt(e.target.value) })
              }
            />
          </label>
        </>
      )}

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
