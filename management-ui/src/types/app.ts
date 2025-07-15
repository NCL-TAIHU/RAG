// src/types/app.ts

export type RouterConfig = {
    type: "simple";
  };
  
  export type RerankerConfig = {
    type: "identity" | "auto_model";
  };
  
  // -------- Embedder --------
  export type AutoModelEmbedderConfig = {
    type: "auto_model";
    embedding_type: "dense";
    model_name: string;
  };
  
  export type BGEEmbedderConfig = {
    type: "bge";
    embedding_type: "sparse";
    model_name: string;
  };
  
  export type EmbedderConfig = AutoModelEmbedderConfig | BGEEmbedderConfig;
  
  // -------- Chunker --------
  export type LengthChunkerConfig = {
    type: "length_chunker";
    chunk_size: number;
    overlap: number;
  };
  
  export type SentenceChunkerConfig = {
    type: "sentence_chunker";
    language: "en" | "zh";
  };
  
  export type ChunkerConfig = LengthChunkerConfig | SentenceChunkerConfig;
  
  // -------- Vector Set --------
  export type VectorSetConfig = {
    root: string;
    dataset: string;
    channel: string;
    chunker: ChunkerConfig;
    embedder: EmbedderConfig;
  };
  
  // -------- Search Engines --------
  export type MilvusConfig = {
    type: "milvus";
    vector_set: VectorSetConfig;
  };
  
  export type ElasticSearchConfig = {
    type: "elasticsearch";
    dataset: string;
    es_host: string;
    es_index: string;
  };
  
  export type HybridMilvusConfig = {
    type: "hybrid_milvus";
    sparse_vector_set: VectorSetConfig;
    dense_vector_set: VectorSetConfig;
    alpha: number;
  };
  
  export type SequentialConfig = {
    type: "sequential";
    engines: SearchEngineConfig[];
  };
  
  export type SearchEngineConfig =
    | MilvusConfig
    | ElasticSearchConfig
    | HybridMilvusConfig
    | SequentialConfig;
  
  // -------- App Metadata --------
  export type AppConfig = {
    id: string;
    name: string;
    dataset: string;
    description?: string;
  
    search_engines: SearchEngineConfig[];
    router: RouterConfig;
    reranker: RerankerConfig;
    max_files?: number;
  
    weave_url?: string;
    created_by?: string;
    created_at?: string;
    updated_at?: string;
  };
  