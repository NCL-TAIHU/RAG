// src/components/forms/SearchEngineForm.tsx

import { useState } from "react";
import type {
  VectorManagerConfig,
  SearchEngineConfig,
  MilvusConfig,
  HybridMilvusConfig,
  ElasticSearchConfig,
} from "../../../types/app";
import { SEARCH_ENGINE_TYPES, DEFAULT_ENGINE_CONFIGS } from "../../../lib/config";

interface Props {
  dataset: string;
  vectorSets: VectorManagerConfig[];
  searchEngines: SearchEngineConfig[];
  onChange: (engines: SearchEngineConfig[]) => void;
}

export default function SearchEngineForm({ dataset, vectorSets, searchEngines, onChange }: Props) {
  const [engineType, setEngineType] = useState<"milvus" | "hybrid_milvus" | "elasticsearch">("milvus");

  const [vectorType, setVectorType] = useState<"sparse" | "dense">("dense");
  const [selectedVector, setSelectedVector] = useState<number>(0);

  const [selectedSparse, setSelectedSparse] = useState<number>(0);
  const [selectedDense, setSelectedDense] = useState<number>(0);

  const [esHost, setEsHost] = useState(DEFAULT_ENGINE_CONFIGS.elasticsearch.es_host);
  const [esIndex, setEsIndex] = useState(DEFAULT_ENGINE_CONFIGS.elasticsearch.es_index);

  const handleAdd = () => {
    let config: SearchEngineConfig;

    if (engineType === "milvus") {
      const vec = vectorSets[selectedVector];
      const milvus: MilvusConfig = {
        type: "milvus",
        dataset,
        vector_type: vectorType,
        vector_manager: vec,
      };
      config = milvus;
    } else if (engineType === "hybrid_milvus") {
      const sparse = vectorSets[selectedSparse];
      const dense = vectorSets[selectedDense];
      const hybrid: HybridMilvusConfig = {
        type: "hybrid_milvus",
        dataset,
        sparse_vector_manager: sparse,
        dense_vector_manager: dense,
        alpha: DEFAULT_ENGINE_CONFIGS.hybrid_milvus.alpha,
      };
      config = hybrid;
    } else {
      const elastic: ElasticSearchConfig = {
        type: "elasticsearch",
        dataset,
        es_host: esHost,
        es_index: esIndex,
      };
      config = elastic;
    }

    onChange([...searchEngines, config]);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <label>
        Search Engine Type
        <select value={engineType} onChange={(e) => setEngineType(e.target.value as any)}>
          {SEARCH_ENGINE_TYPES.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </label>

      {engineType === "milvus" && (
        <>
          <label>
            Vector Type
            <select value={vectorType} onChange={(e) => setVectorType(e.target.value as any)}>
              <option value="dense">dense</option>
              <option value="sparse">sparse</option>
            </select>
          </label>

          <label>
            Vector Set
            <select value={selectedVector} onChange={(e) => setSelectedVector(Number(e.target.value))}>
              {vectorSets.map((vs, i) => (
                <option key={i} value={i}>
                  {vs.channel} • {vs.embedder.type} / {vs.chunker.type}
                </option>
              ))}
            </select>
          </label>
        </>
      )}

      {engineType === "hybrid_milvus" && (
        <>
          <label>
            Sparse Vector Set
            <select value={selectedSparse} onChange={(e) => setSelectedSparse(Number(e.target.value))}>
              {vectorSets.map((vs, i) => (
                <option key={i} value={i}>
                  {vs.channel} • {vs.embedder.type} / {vs.chunker.type}
                </option>
              ))}
            </select>
          </label>

          <label>
            Dense Vector Set
            <select value={selectedDense} onChange={(e) => setSelectedDense(Number(e.target.value))}>
              {vectorSets.map((vs, i) => (
                <option key={i} value={i}>
                  {vs.channel} • {vs.embedder.type} / {vs.chunker.type}
                </option>
              ))}
            </select>
          </label>
        </>
      )}

      {engineType === "elasticsearch" && (
        <>
          <label>
            ElasticSearch Host
            <input type="text" value={esHost} onChange={(e) => setEsHost(e.target.value)} />
          </label>

          <label>
            Index Name
            <input type="text" value={esIndex} onChange={(e) => setEsIndex(e.target.value)} />
          </label>
        </>
      )}

      <button onClick={handleAdd}>Add Search Engine</button>

      {searchEngines.length > 0 && (
        <div>
          <h4>Configured Engines</h4>
          <ul>
            {searchEngines.map((se, i) => (
              <li key={i}>{se.type}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
