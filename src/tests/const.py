from src.core.schema import VectorSetConfig, LengthChunkerConfig, AutoModelEmbedderConfig

ID = "ncl_test"
NCL_DENSE_VS = VectorSetConfig(
    id=ID,
    root=f"tests/storage/vector_set/{ID}", 
    dataset="ncl", 
    channel="abstract_chinese", 
    chunker=LengthChunkerConfig(
        type="length_chunker",
        chunk_size=512,
        overlap=50
    ), 
    embedder=AutoModelEmbedderConfig(
        type="auto_model",
        embedding_type="dense",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)