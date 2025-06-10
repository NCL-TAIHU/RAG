from vllm import LLM, SamplingParams
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

class LLMBuilder:
    """
    Builder class for creating an LLM instance.
    """
    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod    
    def from_default(cls, model_name):
        llm_config = LLMConfig(
            model= model_name, 
            trust_remote_code=True,
            download_dir="/tmp/model_cache",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # 限制 GPU 記憶體使用率
            max_model_len=8192,          # 限制上下文長度
            quantization="fp8"
        )
        return cls(config=llm_config)
    
    def build(self) -> LLM:
        """
        Builds and returns an LLM instance based on the provided configuration.
        """
        try: 
            return LLM(**self.config.model_dump(exclude_none=True))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
        

def generate(llm: LLM, prompt: str) -> str: 
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    response = llm.generate(prompt, sampling_params=params)
    return response[0].outputs[0].text if response and response[0].outputs else "No response generated."
