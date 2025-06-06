from typing import List

class PromptBuilder():
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.user_prompt = ""
        self.retrieval_results = ""
        self.assistant_prompt = ""
        self.history = []

    @classmethod
    def from_default(cls, name):
        """使用默認系統提示創建一個新的 PromptBuilder 實例"""
        default_system_prompt = "You are a helpful assistant."
        return cls(system_prompt=default_system_prompt)
    
    def add_user_message(self, message: str):
        """添加用戶消息到提示中"""
        self.user_prompt += message + ','
        self.history.append(f"User: {message}")
        return self
    
    def add_assistant_message(self, message: str):
        """添加助手消息到提示中"""
        self.assistant_prompt += message + ','
        self.history.append(f"Assistant: {message}")
        return self
    
    def add_retrieval_results(self, results: str):
        """添加檢索結果到提示中"""
        self.retrieval_results += results + ','
        self.history.append(f"Retrieval Results: {results}")
        return self
    
    def add_documents(self, documents: List[str]):
        self.retrieval_results += ', '.join(documents) + ','
        self.history.append(f"Documents: {', '.join(documents)}")
        return self
    
    def build_prompt(self) -> str:
        """構建最終的提示字符串"""
        prompt = ""
        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n"
        if self.user_prompt:
            prompt += f"User: {self.user_prompt} \n"
        if self.assistant_prompt:
            prompt += f"Assistant: {self.assistant_prompt} \n"
        if self.retrieval_results:
            prompt += f"Retrieval Results: {self.retrieval_results} \n"
        return prompt.strip()