// src/lib/config.ts

export const AVAILABLE_DATASETS = [
    "ncl",
    "litsearch"
];
  
export const EMBEDDER_MODELS = {
bge: ["BAAI/bge-m3"],
auto_model: [
    "sentence-transformers/all-MiniLM-L6-v2"
]
};

export const CHUNKERS = [
{
    type: "length_chunker",
    default: { chunk_size: 512, overlap: 50 }
},
{
    type: "sentence_chunker",
    default: {language: "en"}
}
];

export const SEARCH_ENGINE_TYPES = [
"milvus",
"elasticsearch",
"hybrid_milvus",
"sequential"
];

export const DEFAULT_ENGINE_CONFIGS = {
milvus: {
    vector_type: "dense",
    alpha: 0.5,
},
hybrid_milvus: {
    alpha: 0.5
},
elasticsearch: {
    es_host: "http://localhost:9200",
    es_index: "default-index"
},
sequential: {
    engines: []
}
};
