from src.core.schema import VectorSetConfig, ChunkerConfig, LengthChunkerConfig, EmbedderConfig, AutoModelEmbedderConfig
from src.core.vector_set import BaseVectorSet
from src.run.state import BaseState
from src.tests.const import NCL_DENSE_VS
import logging
from src.utils.logging import setup_logger

logger = setup_logger("vector_set_test", console=True, file=False, level=logging.DEBUG)

CONFIG_DIR = "tests/configs/vector_set"
state = BaseState[VectorSetConfig, BaseVectorSet](
    config_cls=VectorSetConfig,
    obj_cls=BaseVectorSet,
    config_dir=CONFIG_DIR
) 

config = NCL_DENSE_VS
state.register(config)
id = state.get_config(NCL_DENSE_VS.id).id
state.activate(id)
