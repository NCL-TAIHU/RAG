from vllm import LLM
from pydantic import BaseModel
from typing import Optional

class LLMConfig(BaseModel):
    """
    Configuration for the LLMBuilder.
    """
    model: str  # required
    trust_remote_code: Optional[bool] = None
    download_dir: Optional[str] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None

def get_llm(config: LLMConfig) -> LLM:
    try:
        llm = LLM(**config.model_dump(exclude_none=True))
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {str(e)}")