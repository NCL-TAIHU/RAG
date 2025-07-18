"""
TODO:
- End to end response time test
- Multiple user requests
- Resource trace
"""

from src.tests.const import APPS
from src.run.state import BaseState
from src.core.app import App
from pprint import pprint
from src.core.schema import AppConfig
from src.core.filter import Filter
from src.utils.logging import setup_logger
import logging

logger = setup_logger("taihu", console=True, file=False, level=logging.DEBUG)

app_state = BaseState[AppConfig, App](
    config_cls=AppConfig,
    obj_cls=App,
    config_dir="_tests/configs/app"
)

def filter_app(app_config: AppConfig) -> bool:
    return (
        app_config.dataset == "ncl" and
        len(app_config.search_engines) > 0 and
        app_config.router.type == "simple" and
        app_config.reranker.type == "identity" and
        app_config.search_engines[0].type == "milvus" and
        app_config.search_engines[0].vector_set.chunker.type == "length_chunker" and
        app_config.search_engines[0].vector_set.embedder.type == "auto_model"
    )

filtered_apps = [conf for conf in APPS if filter_app(conf)]

for app_config in filtered_apps:
    app_state.register(app_config)
    logger.info(f"Registering app: {app_config.name} with ID {app_config.id}")

# activate 

registered_apps = app_state.list_ids()
target_app_id = registered_apps[0]
logger.info(f"Activating app: \n {app_state.get_config(target_app_id)}")

# test end to end response time for app.
import time
start_time = time.time()
app_state.activate(target_app_id)
end_time = time.time()
logger.info(f"App {target_app_id} activated in {end_time - start_time:.2f} seconds.")

#test search time
app = app_state.get_obj(target_app_id)
app_config = app_state.get_config(target_app_id)
filt_cls = Filter.from_dataset(app_config.dataset)
filt = filt_cls()
search_query = "What is the capital of France?"
start_time = time.time()
results = app.search(query=search_query, filter=filt, limit=5)
end_time = time.time()
logger.info(f"Search completed in {end_time - start_time:.2f} seconds.")
#app_with_id --> register --> activate --> get_obj