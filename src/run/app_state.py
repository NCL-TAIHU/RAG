import os
import json
from typing import Dict, List, Optional, Any
from src.core.app import SearchApp
from src.core.schema import AppConfig  # Your unified config schema
from src.core.document import Document
import yaml
from pydantic import ValidationError


config = yaml.safe_load(open("config/apps.yml", "r", encoding="utf-8"))
APP_METADATA_DIR = config["path"]

class AppState:
    """
    The grand state of the system. Model as in MVC design pattern.
    Owns:
    - Disk-backed app metadata (configs)
    - In-memory SearchApp instances
    - Benchmark definitions (TODO)
    - Weave URLs (from metadata)
    """
    def __init__(self):
        self._apps: Dict[str, SearchApp] = {}
        self._configs: Dict[str, AppConfig] = {}
    
    def load_all_metadata(self):
        if not os.path.exists(APP_METADATA_DIR):
            os.makedirs(APP_METADATA_DIR)

        for fname in os.listdir(APP_METADATA_DIR):
            if fname.endswith(".yml") or fname.endswith(".yaml"):
                path = os.path.join(APP_METADATA_DIR, fname)
                with open(path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
                    try:
                        config = AppConfig.model_validate(raw)
                        self._configs[config.id] = config
                    except ValidationError as e:
                        print(f"[WARN] Skipping invalid metadata file {fname}: {e}")

    def list_apps(self) -> List[str]:
        return list(self._configs.keys())

    def get_config(self, id: str) -> AppConfig:
        return self._configs[id]

    def register_app(self, config: AppConfig):
        self._configs[config.id] = config

        path = os.path.join(APP_METADATA_DIR, f"{config.name}_{config.id}.yml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, sort_keys=False)

    def activate_app(self, id: str):
        config = self._configs[id]
        app = SearchApp.from_config(config)
        app.setup()
        self._apps[id] = app

    def get_app(self, name: str) -> SearchApp:
        return self._apps[name]

    def remove_app(self, name: str):
        if name in self._apps:
            del self._apps[name]
        if name in self._configs:
            del self._configs[name]
        path = os.path.join(APP_METADATA_DIR, f"{name}.yml")
        if os.path.exists(path):
            os.remove(path)