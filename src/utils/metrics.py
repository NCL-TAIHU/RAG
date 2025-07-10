import json
import time
import psutil
import torch
from transformers import AutoTokenizer
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class MetricsTracker:
    def __init__(self, log_file: str = "logs/usage_metrics.jsonl", model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.log_file = log_file
        self.start_time = None
        self.metrics = {}
        self.model_name = model_name
        
        # 初始化 tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_name}, using default tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                trust_remote_code=True,
                use_fast=True
            )
        
        # 確保日誌目錄存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def start_tracking(self):
        """開始追蹤指標"""
        self.start_time = time.time()
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu_usage": self._get_gpu_usage(),
            "memory_usage": self._get_memory_usage(),
            "cpu_usage": self._get_cpu_usage()
        }
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        return len(self.tokenizer.encode(text))
    
    def end_tracking(self, query: str, llm_response: str, prompt: str, results: Optional[Any] = None):
        """結束追蹤並記錄指標"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # 計算 token 數量
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(llm_response)
        total_tokens = prompt_tokens + response_tokens
        
        metrics = {
            **self.metrics,
            "duration": duration,
            "query": query,
            "llm_response": llm_response,
            "prompt": prompt,
            "results": [result.model_dump() if hasattr(result, 'model_dump') else result.dict() if hasattr(result, 'dict') else result for result in results] if results else [],
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens,
                "model": self.model_name
            }
        }
        
        # 將指標寫入 JSONL 檔案
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        
        return metrics
    
    def _get_gpu_usage(self) -> Dict[str, float]:
        """獲取 GPU 使用率"""
        if not torch.cuda.is_available():
            return {"gpu_usage": 0.0}
        
        try:
            gpu = torch.cuda.get_device_properties(0)
            return {
                "gpu_usage": torch.cuda.memory_allocated(0) / gpu.total_memory * 100,  # GPU 記憶體使用百分比 (%)
                "gpu_memory_used": torch.cuda.memory_allocated(0) / 1024**2,  # 已使用的 GPU 記憶體 (MB)
                "gpu_memory_total": gpu.total_memory / 1024**2  # GPU 總記憶體容量 (MB)
            }
        except Exception:
            return {"gpu_usage": 0.0}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """獲取記憶體使用率"""
        memory = psutil.virtual_memory()
        return {
            "memory_usage_percent": memory.percent,  # 系統記憶體使用百分比 (%)
            "memory_used": memory.used / 1024**2,  # 已使用的系統記憶體 (MB)
            "memory_total": memory.total / 1024**2  # 系統總記憶體容量 (MB)
        }
    
    def _get_cpu_usage(self) -> float:
        """獲取 CPU 使用率"""
        return psutil.cpu_percent()  # CPU 使用百分比 (%) 