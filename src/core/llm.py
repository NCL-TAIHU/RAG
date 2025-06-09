from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import Optional
import os
from openai import OpenAI

class OpenAIChatLLM:
    def __init__(self, model_name: str, system_prompt: Optional[str] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



class LLMConfig(BaseModel):
    """
    Configuration for the LLMBuilder.
    """
    model: str  # required
    use_openai: Optional[bool] = False
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
    def from_default(cls, model_name, use_openai: bool = False):
        if use_openai:
            llm_config = LLMConfig(
                model=model_name,
                use_openai=True
            )
            return cls(config=llm_config)
        else:
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
        if self.config.use_openai:
            return OpenAIChatLLM(
                model_name=self.config.model,
                system_prompt="你是一個歷史領域的研究生，你在使用一個基於RAG技術的論文檢索系統。"  
            )
        else:
            try: 
                return LLM(**self.config.model_dump(exclude_none=True))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM: {str(e)}")



def generate(llm, prompt: str):
    """
    Unified generate function for both vLLM and OpenAIChatLLM.
    Returns: List[{"text": str}]
    """
    # case 1: vLLM
    if isinstance(llm, LLM):
        params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
        raw_outputs = llm.generate(prompt, sampling_params=params)
        return [{"text": output.outputs[0].text} for output in raw_outputs if output.outputs]

    # case 2: OpenAIChatLLM
    elif isinstance(llm, OpenAIChatLLM):
        completion = llm.client.chat.completions.create(
            model=llm.model_name,
            messages=[
                {"role": "system", "content": llm.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return [{"text": completion.choices[0].message.content}]

    else:
        raise TypeError(f"Unsupported LLM type: {type(llm)}")

