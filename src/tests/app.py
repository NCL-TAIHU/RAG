from src.core.schema import (
    AppConfig, 
    SearchEngineConfig,
    MilvusConfig, 
    VectorSetConfig, 
    RouterConfig, 
    RerankerConfig
)
from src.core.app import App 
from src.run.state import BaseState
from src.utils.logging import setup_logger
from src.tests.const import NCL_DENSE_VS

import logging

logger = setup_logger("app_test", console=True, file=False, level=logging.DEBUG)

CONFIG_DIR = "_tests/configs/app"
state = BaseState[AppConfig, App](
    config_cls=AppConfig,
    obj_cls=App,
    config_dir=CONFIG_DIR
) 

ID = "test_app"
config = AppConfig(
    id=ID,
    name="Test App",
    dataset = "ncl",
    description="A test application for vector set operations",
    search_engines=[MilvusConfig(
        type="milvus", 
        vector_set=NCL_DENSE_VS
    )], 
    router = RouterConfig(type="simple"), 
    reranker = RerankerConfig(type="identity")
)

state.register(config)
id = state.get_config(config.id).id
state.activate(id)