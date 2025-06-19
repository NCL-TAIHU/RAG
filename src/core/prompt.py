from typing import List
from src.core.entity import Document

class PromptBuilder():
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or """你是一個專業的學術研究助手。你的任務是分析文獻並提供有價值的見解。請確保回答簡潔、完整且不重複。"""
        self.user_prompt = ""
        self.retrieval_results = ""
        self.assistant_prompt = ""
        self.history = []

    @classmethod
    def from_default(cls, name):
        """使用默認系統提示創建一個新的 PromptBuilder 實例"""
        default_system_prompt = """你是一個專業的學術研究助手。你的任務是分析文獻並提供有價值的見解。請確保回答簡潔、完整且不重複。"""
        return cls(system_prompt=default_system_prompt)
    
    def add_user_message(self, message: str):
        """添加用戶消息到提示中"""
        self.user_prompt += message
        self.history.append(f"User: {message}")
        return self
    
    def add_assistant_message(self, message: str):
        """添加助手消息到提示中"""
        self.assistant_prompt += message
        self.history.append(f"Assistant: {message}")
        return self
    
    def add_retrieval_results(self, results: str):
        """添加檢索結果到提示中"""
        self.retrieval_results += results
        self.history.append(f"Retrieval Results: {results}")
        return self
    
    def add_documents(self, documents: List[Document]):
        """添加文檔到提示中"""
        if not documents:
            return self
        results = []
        for i, doc in enumerate(documents, 1):
            abstract = doc.chinese.abstract if doc.chinese and doc.chinese.abstract else \
                      doc.english.abstract if doc.english and doc.english.abstract else ""
            if abstract:
                results.append(f"[文獻 {i}]\n{abstract}\n")
        self.add_retrieval_results("\n".join(results))
        return self
    
    def build_prompt(self) -> str:
        """構建最終的提示字符串"""
        prompt = ""
        if self.system_prompt:
            prompt += f"System: {self.system_prompt}\n\n"
        if self.user_prompt:
            prompt += f"User: {self.user_prompt}\n\n"
        if self.retrieval_results:
            prompt += f"以下是相關文獻摘要：\n{self.retrieval_results}\n\n"
            prompt += "請依照以下指引生成回答：\n"
            prompt += "- 用一段文字統整這些論文可能涉及的核心研究主題與趨勢，避免逐一重複每篇內容。\n"
            prompt += "- 請在這段文字中自然地帶出 2 到 3 個可能的後續研究方向。\n\n"
            prompt += "請注意：\n"
            prompt += "- 不要說『本文』或『以上回應』，直接給出分析結果。\n"
            prompt += "- 不要使用條列格式、標題、步驟編號、格式化開頭（如『本文將…』）。\n"
            prompt += "- 保持回答的完整性，確保有明確的結尾。\n"
            prompt += "- 避免重複內容，確保回答簡潔。\n"
            prompt += "- 不要使用『綜上所述』等總結性詞語。\n\n"
        if self.assistant_prompt:
            prompt += f"Assistant: {self.assistant_prompt}\n"
        return prompt.strip()