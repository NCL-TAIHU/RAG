// src/lib/config.ts

export const AVAILABLE_DATASETS = [
    "ncl",
    "litsearch"
  ];
  
export const EMBEDDER_MODELS = {
bge: ["BAAI/bge-m3"],
auto_model: [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-small-v2"
]
};

export const CHUNKERS = [
{
    type: "simple_chunker",
    default: { chunk_size: 512, overlap: 50 }
},
{
    type: "sentence_chunker",
    default: { max_length: 512, stride: 50 }
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
