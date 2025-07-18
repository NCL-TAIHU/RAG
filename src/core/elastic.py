from pydantic import BaseModel
from elasticsearch import Elasticsearch
import os
import json
from typing import Dict
import logging

logger = logging.getLogger('taihu')

class ElasticIndexConfig(BaseModel):
    es_index: str
    fields: Dict[str, str]  # e.g., {"year": "integer", "author": "keyword"}


class ElasticIndexBuilder:
    def __init__(self, es: Elasticsearch, config: ElasticIndexConfig):
        self.es = es
        self.config = config
        self.config_path = os.path.join("db", "es_configs", f"{config.es_index}.json")
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

    def _config_matches_saved(self) -> bool:
        if not os.path.exists(self.config_path):
            return False
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                saved = ElasticIndexConfig(**json.load(f))
            return saved.model_dump() == self.config.model_dump()
        except Exception as e:
            print(f"[ERROR] Config comparison failed: {e}")
            return False

    def _save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2)

    def build(self, force_rebuild: bool = False):
        if self.es.indices.exists(index=self.config.es_index):
            if force_rebuild or not self._config_matches_saved():
                logger.info(f"Rebuilding index {self.config.es_index}")
                self.es.indices.delete(index=self.config.es_index)
            else:
                logger.info(f"Reusing existing index {self.config.es_index}")
                return
        
        mapping = {
            "mappings": {
                "properties": {
                    field: {"type": dtype}
                    for field, dtype in self.config.fields.items()
                }
            }
        }
        self.es.indices.create(index=self.config.es_index, body=mapping)
        self._save_config()