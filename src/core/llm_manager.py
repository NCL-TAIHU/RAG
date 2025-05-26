from vllm import LLM, SamplingParams
from loguru import logger
import os
from huggingface_hub import login
from transformers import AutoConfig
import torch
from datetime import datetime
import csv
import json

class LLMManager:
    """LLM管理類別，負責處理大型語言模型的互動"""
    
    def __init__(self, model_name: str):
        """初始化LLM管理器
        
        Args:
            model_name: 模型名稱
        """
        try:
            # 設置 GPU 記憶體優化
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # 確保 HF Token 已設置
            token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not token:
                raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment")
            
            # 清除可能衝突的環境變數
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            
            # 登入 Hugging Face
            login(token=token)
            
            # 下載模型配置
            logger.info(f"Downloading model config for {model_name}")
            config = AutoConfig.from_pretrained(model_name, token=token)
            logger.info(f"Model config downloaded successfully")
            
            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            
            # 初始化 vLLM with 記憶體優化設定
            self.model = LLM(
                model=model_name,
                trust_remote_code=True,
                download_dir="/tmp/model_cache",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # 限制 GPU 記憶體使用率
                max_model_len=8192,          # 限制上下文長度
                quantization="fp8"
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512
            )
            logger.info(f"LLM initialized with model: {model_name}")
            
            # 設置輸出目錄
            self.output_dir = "outputs"
            os.makedirs(self.output_dir, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def generate(self, prompt: str) -> str:
        """生成文本回應
        
        Args:
            prompt: 輸入提示詞
            
        Returns:
            str: 生成的回應文本
        """
        try:
            # 清理 GPU 記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            outputs = self.model.generate(prompt, self.sampling_params)
            if not outputs or not outputs[0].outputs:
                logger.warning("No output generated from LLM")
                return ""
            
            # 獲取原始回應
            raw_response = outputs[0].outputs[0].text
            logger.info("Raw LLM response:")
            logger.info("="*50)
            logger.info(raw_response)
            logger.info("="*50)
            
            # 儲存回應
            self.save_response(prompt, raw_response)
            
            return raw_response
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            raise
            
    def _clean_response(self, response: str) -> str:
        """清理和格式化 LLM 回應
        
        Args:
            response: 原始回應文本
            
        Returns:
            str: 清理後的回應文本
        """
        try:
            # 移除多餘的空行和空白
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # 移除常見的無關前綴
            prefixes_to_remove = [
                "對於可能的後續研究方向，給出清晰簡要的描述，不要過於冗長或複雜。",
                "答案：",
                "回答：",
                "以下是",
                "這是",
            ]
            
            # 找到實際內容的起始位置
            start_idx = 0
            for i, line in enumerate(lines):
                # 檢查是否為無關前綴
                if any(line.startswith(prefix) for prefix in prefixes_to_remove):
                    start_idx = i + 1
                    continue
                # 如果行太短或看起來像是標題，跳過
                if len(line) < 10 or line.endswith('：'):
                    start_idx = i + 1
                    continue
                break
            
            # 獲取清理後的內容
            cleaned_lines = lines[start_idx:]
            if not cleaned_lines:
                return response  # 如果清理後沒有內容，返回原始回應
            
            # 組合清理後的文本
            cleaned_text = '\n'.join(cleaned_lines)
            
            return cleaned_text.strip()
            
        except Exception as e:
            logger.error(f"Error in cleaning response: {str(e)}")
            return response  # 如果清理失敗，返回原始回應

    def save_response(self, prompt: str, response: str):
        """儲存生成的回應到檔案
        
        Args:
            prompt: 輸入提示詞
            response: 生成的回應
        """
        try:
            # 生成時間戳記
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 儲存到 CSV
            csv_file = os.path.join(self.output_dir, f"llm_responses_{timestamp}.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Prompt', 'Response'])
                writer.writerow([timestamp, prompt, response])
            
            # 儲存到 TXT
            txt_file = os.path.join(self.output_dir, f"llm_response_{timestamp}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Prompt:\n{prompt}\n\n")
                f.write("Generated Response:\n")
                f.write(response)
                
            logger.info(f"Response saved to {csv_file} and {txt_file}")
            
        except Exception as e:
            logger.error(f"Error saving response: {str(e)}")
            # 不要在這裡 raise，以免影響主要功能 