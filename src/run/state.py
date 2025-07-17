from typing import TypeVar, Generic, Dict, Type, List
from src.core.interface import StoredObj, StoredConfig
from pydantic import BaseModel, ValidationError
import os, yaml

T = TypeVar('T', bound=StoredConfig)
S = TypeVar('S', bound=StoredObj)

class BaseState(Generic[T, S]):
    def __init__(self, config_cls: Type[T], obj_cls: Type[S], config_dir: str):
        self._configs: Dict[str, T] = {}
        self._objs: Dict[str, S] = {}
        self._config_dir = config_dir

        #class defs
        self._config_cls = config_cls
        self._obj_cls = obj_cls
    
    def load_all_configs(self):
        if not os.path.exists(self._config_dir):
            os.makedirs(self._config_dir)

        for fname in os.listdir(self._config_dir):
            if fname.endswith(".yml") or fname.endswith(".yaml"):
                path = os.path.join(self._config_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        raw = yaml.safe_load(f)
                        config = self._config_cls.model_validate(raw)
                        self._configs[config.id] = config
                    except ValidationError as e:
                        print(f"[WARN] Skipping invalid config file {fname}: {e}")

    def list_apps(self) -> List[str]:
        return list(self._configs.keys())

    def get_config(self, id: str) -> T:
        return self._configs[id]

    def register(self, config: T):
        self._configs[config.id] = config

        path = os.path.join(self._config_dir, f"{config.id}.yml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, sort_keys=False)

    def activate(self, id: str):
        config = self._configs[id]
        obj = self._obj_cls.from_config(config)
        obj.setup()
        self._objs[id] = obj

    def get_app(self, name: str) -> S:
        return self._objs[name]

    def remove_app(self, id: str):
        if id in self._objs:
            del self._objs[id]
        if id in self._configs:
            del self._configs[id]
        path = os.path.join(self._config_dir, f"{id}.yml")
        if os.path.exists(path):
            os.remove(path)