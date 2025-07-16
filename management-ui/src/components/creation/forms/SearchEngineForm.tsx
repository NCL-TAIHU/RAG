// src/components/forms/SearchEngineForm.tsx

import { useState } from "react";
import type {
  VectorSetConfig,
  SearchEngineConfig,
  MilvusConfig,
  HybridMilvusConfig,
  ElasticSearchConfig,
} from "../../../types/app";
import { SEARCH_ENGINE_TYPES, DEFAULT_ENGINE_CONFIGS } from "../../../lib/config";

interface Props {
  dataset: string;
  vectorSets: VectorSetConfig[];
  searchEngines: SearchEngineConfig[];
  onChange: (engines: SearchEngineConfig[]) => void;
}

export default function SearchEngineForm({ dataset, vectorSets, searchEngines, onChange }: Props) {
  const [engineType, setEngineType] = useState<"milvus" | "hybrid_milvus" | "elasticsearch">("milvus");

  // Shared vector selectors
  const [vectorType, setVectorType] = useState<"sparse" | "dense">("dense");
  const [selectedVectorIndex, setSelectedVectorIndex] = useState<number>(0);
  const [selectedSparseIndex, setSelectedSparseIndex] = useState<number>(0);
  const [selectedDenseIndex, setSelectedDenseIndex] = useState<number>(0);

  // ElasticSearch fields
  const [esHost, setEsHost] = useState(DEFAULT_ENGINE_CONFIGS.elasticsearch.es_host);
  const [esIndex, setEsIndex] = useState(DEFAULT_ENGINE_CONFIGS.elasticsearch.es_index);

  const renderVectorSetOptions = (
    selectedIndex: number,
    onChange: (index: number) => void,
    filter?: (vs: VectorSetConfig) => boolean
  ) => {
    const filteredVectorSets = vectorSets
      .map((vs, i) => ({ vs, i }))
      .filter(({ vs }) => (filter ? filter(vs) : true));

    return (
      <select
        value={selectedIndex}
        onChange={(e) => onChange(Number(e.target.value))}
      >
        {filteredVectorSets.map(({ vs, i }) => (
          <option key={i} value={i}>
            {vs.channel} • {vs.embedder.type} / {vs.chunker.type}
          </option>
        ))}
      </select>
    );
  };


  const handleAdd = () => {
    let config: SearchEngineConfig;

    if (engineType === "milvus") {
      const milvusConfig: MilvusConfig = {
        type: "milvus",
        vector_set: vectorSets[selectedVectorIndex],
      };
      config = milvusConfig;
    } else if (engineType === "hybrid_milvus") {
      const hybridConfig: HybridMilvusConfig = {
        type: "hybrid_milvus",
        sparse_vector_set: vectorSets[selectedSparseIndex],
        dense_vector_set: vectorSets[selectedDenseIndex],
        alpha: DEFAULT_ENGINE_CONFIGS.hybrid_milvus.alpha,
      };
      config = hybridConfig;
    } else {
      const elasticConfig: ElasticSearchConfig = {
        type: "elasticsearch",
        dataset,
        es_host: esHost,
        es_index: esIndex,
      };
      config = elasticConfig;
    }

    onChange([...searchEngines, config]);
  };


  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Engine Type */}
      <label>
        Search Engine Type
        <select value={engineType} onChange={(e) => setEngineType(e.target.value as any)}>
          {SEARCH_ENGINE_TYPES.map((type) => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </label>

      {/* Milvus Config */}
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
            {renderVectorSetOptions(selectedVectorIndex, setSelectedVectorIndex)}
          </label>
        </>
      )}

      {/* Hybrid Milvus Config */}
      {engineType === "hybrid_milvus" && (
        <>
          <label>
            Sparse Vector Set
            {renderVectorSetOptions(
              selectedSparseIndex,
              setSelectedSparseIndex,
              (vs) => vs.embedder.embedding_type === "sparse"
            )}
          </label>

          <label>
            Dense Vector Set
            {renderVectorSetOptions(
              selectedDenseIndex,
              setSelectedDenseIndex,
              (vs) => vs.embedder.embedding_type === "dense"
            )}
          </label>
        </>
      )}


      {/* ElasticSearch Config */}
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

      {/* Display Configured Engines */}
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