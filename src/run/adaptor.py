"""
Adaptor functions that hydrate form-based dictionaries into validated backend AppConfig models.
Injects backend-only state like vector store roots, weave runs, and timestamps.
"""

from typing import Dict, Any, List
from datetime import datetime
from uuid import uuid4
import weave

from src.core.schema import (
    VectorSetConfig,
    AppConfig,
    SearchEngineConfig,
    MilvusConfig,
    HybridMilvusConfig,
    ElasticSearchConfig,
    SequentialConfig,
)

VECTOR_STORE_ROOT = "/mnt/vector_stores"


def hydrate_vector_set(form: Dict[str, Any]) -> VectorSetConfig:
    """
    Enrich a vector set form dict with backend-computed root path.
    """
    dataset = form["dataset"]
    channel = form["channel"]
    chunker_type = form["chunker"]["type"]
    form["root"] = f"{VECTOR_STORE_ROOT}/{dataset}/{channel}/{chunker_type}"
    return VectorSetConfig(**form)


def hydrate_search_engine(form: Dict[str, Any]) -> SearchEngineConfig:
    """
    Recursively hydrate a search engine config with enriched vector sets and validations.
    """
    engine_type = form["type"]

    if engine_type == "milvus":
        form["vector_set"] = hydrate_vector_set(form["vector_set"]).dict()
        return MilvusConfig(**form)

    elif engine_type == "hybrid_milvus":
        form["sparse_vector_set"] = hydrate_vector_set(form["sparse_vector_set"]).dict()
        form["dense_vector_set"] = hydrate_vector_set(form["dense_vector_set"]).dict()

        sparse_type = form["sparse_vector_set"]["embedder"]["embedding_type"]
        dense_type = form["dense_vector_set"]["embedder"]["embedding_type"]

        if sparse_type != "sparse":
            raise ValueError("HybridMilvusConfig requires sparse_vector_set.embedder.embedding_type == 'sparse'")
        if dense_type != "dense":
            raise ValueError("HybridMilvusConfig requires dense_vector_set.embedder.embedding_type == 'dense'")

        return HybridMilvusConfig(**form)

    elif engine_type == "elasticsearch":
        return ElasticSearchConfig(**form)

    elif engine_type == "sequential":
        form["engines"] = [hydrate_search_engine(e).model_dump() for e in form["engines"]]
        return SequentialConfig(**form)

    else:
        raise ValueError(f"Unsupported search engine type: {engine_type}")


def assemble_app_config(form: Dict[str, Any], created_by: str) -> AppConfig:
    """
    Assemble a full AppConfig from a form dict, injecting IDs, timestamps, vector roots, and weave metadata.
    """
    now = datetime.now(datetime.timezone.utc).isoformat()
    app_id = str(uuid4())

    run = weave.init(name=form["name"])
    weave_url = f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"

    hydrated_engines: List[SearchEngineConfig] = [
        hydrate_search_engine(raw) for raw in form["search_engines"]
    ]

    return AppConfig(
        id=app_id,
        name=form["name"],
        dataset=form["dataset"],
        description=form.get("description"),
        router=form["router"],
        reranker=form["reranker"],
        search_engines=hydrated_engines,
        max_files=form.get("max_files", 1_000_000),
        weave_url=weave_url,
        created_by=created_by,
        created_at=now,
        updated_at=now,
    )
