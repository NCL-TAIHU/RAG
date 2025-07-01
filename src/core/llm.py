from vllm import LLM, SamplingParams
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import requests
import os
from dotenv import load_dotenv
load_dotenv()

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

    @classmethod
    def from_default(cls, model_name: str) -> "LLMConfig":
        """
        Creates a default LLMConfig instance with the specified model name.
        """
        if model_name == 'meta-llama/Llama-3.1-8B-Instruct': 
            return cls(
                model=model_name,
                trust_remote_code=True,
                download_dir="/tmp/model_cache",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # Limit GPU memory utilization
                max_model_len=8192,          # Limit context length
                quantization="fp8"
            )
        elif model_name == 'taide/TAIDE-LX-7B':
            return cls(
                model=model_name,
                trust_remote_code=True,
                download_dir="/tmp/model_cache",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # Limit GPU memory utilization
                max_model_len=4096,          # Limit context length
                quantization="fp8"
            )
        elif model_name == 'taide/Llama-3.1-TAIDE-LX-8B-Chat':
            return cls(
                model=model_name,
                trust_remote_code=True,
                download_dir="/tmp/model_cache",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # Limit GPU memory utilization
                max_model_len=8192,          # Limit context length
                quantization="fp8"
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Please provide a valid model name.")

class Agent: 
    """
    A base class for agents that interact with LLMs (Large Language Models).
    """
    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the provided prompt.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def from_vllm(cls, model_name: str) -> "Agent":
        """
        Creates an Agent instance from a VLLM configuration.
        """
        config = LLMConfig.from_default(model_name=model_name)
        llm = LLM(**config.model_dump(exclude_none=True))
        return VLLMAgent(llm)
    
    @classmethod
    def from_openai(cls, model_name: str) -> "Agent":
        """
        Creates an Agent instance from an OpenAI configuration.
        """
        llm = model_name
        return OpenAIAgent(llm)
    @classmethod
    def from_TaiwanAIRAP(cls, model_name: str) -> "Agent":
        """
        Creates an Agent instance from Taiwan AI RAP API.
        """
        llm = model_name
        return TaiwanAIRAPAgent(llm)


class OpenAIAgent(Agent): 
    """
    An agent that uses OpenAI API to generate responses.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sampling_params = {
            "max_tokens": 20,
            "temperature": 0
        }

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the provided prompt.
        """
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

       
        response = client.chat.completions.create(
            model=self.llm,  
            messages=[
                {"role": "system", "content": "你是一個博覽各研究領域論文的熱心助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.sampling_params["max_tokens"],
            temperature=self.sampling_params["temperature"]
        )
        
        return response.choices[0].message.content if response and response.choices[0].message else "No response generated."   


class VLLMAgent(Agent):
    """
    An agent that uses VLLM to generate responses.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the provided prompt.
        """
        params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
        response = self.llm.generate(prompt, sampling_params=params)
        return response[0].outputs[0].text if response and response[0].outputs else "No response generated."   

class TaiwanAIRAPAgent(Agent): 
    """
    An agent that uses Taiwan AI RAP to generate responses.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
        self.sampling_params = {
            "max_tokens": 20,
            "temperature": 0
        }

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM based on the provided prompt.
        """
        # API URL
        url = "https://portal.genai.nchc.org.tw/api/v1/chat/completions"

        # 請求內容
        payload = {
            "model": self.llm,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一個博覽各研究領域論文的熱心助手。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.sampling_params["max_tokens"],
            "temperature": self.sampling_params["temperature"]
        }

        # 標頭資訊
        headers = {
            "x-api-key": os.getenv("TAIWANAIRAP_API_KEY"),
            "Content-Type": "application/json"
        }

        # 發送 POST 請求
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            print("Token 使用量：", result["usage"])
            return reply
            
        else:
            print("請求失敗，狀態碼：", response.status_code)
            return response.text  
   