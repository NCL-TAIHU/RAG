from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator
import os
import json
from dataclasses import dataclass
from pydantic import BaseModel
from src.core.document import Document, NCLDocument, Info, LitSearchDocument, NCL_LLM_SummaryDocument
import yaml
import openai
import re

@dataclass
class PreprocessConfig:
    """Simple configuration for preprocessing"""
    input_path: str
    sample_size: int = 20
    max_files: int = 5
    encoding: str = "utf-8"

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors"""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.field_mappings = {}
    
    @abstractmethod
    def extract_fields(self, raw_data: Dict) -> Dict:
        """Extract and map fields from raw data"""
        pass
    
    @abstractmethod
    def fill_content(self, extracted_fields: Dict) -> Dict:
        """Fill content structure based on extracted fields"""
        pass
    
    @abstractmethod
    def preprocess_data(self, content: Dict) -> Optional[Document]:
        """Preprocess content and create Document"""
        pass
    
    def process_record(self, raw_data: Dict) -> Optional[Document]:
        """Main processing pipeline: extract -> fill -> preprocess -> document"""
        try:
            # Step 1: Extract fields
            extracted_fields = self.extract_fields(raw_data)
            
            # Step 2: Fill content structure
            content = self.fill_content(extracted_fields)
            
            # Step 3: Preprocess and create document
            document = self.preprocess_data(content)
            
            return document
            
        except Exception as e:
            print(f"Error processing record: {e}")
            return None
    
    def analyze_structure(self) -> Dict:
        """Analyze data structure for field mapping"""
        structure = {}
        sample_count = 0
        
        for file_path in self._get_input_files():
            if sample_count >= self.config.sample_size:
                break
                
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                for line in f:
                    if sample_count >= self.config.sample_size:
                        break
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self._update_structure(data, structure)
                            sample_count += 1
                        except:
                            continue
        
        return structure
    
    def _update_structure(self, data: Dict, structure: Dict):
        """Update structure analysis with one record"""
        for key, value in data.items():
            if key not in structure:
                structure[key] = {
                    'type': type(value).__name__,
                    'has_chinese': False,
                    'has_english': False,
                    'sample': str(value)[:50]
                }
            
            # Detect language
            if isinstance(value, str):
                if any('\u4e00' <= char <= '\u9fff' for char in value):
                    structure[key]['has_chinese'] = True
                if any(c.isalpha() for c in value):
                    structure[key]['has_english'] = True
    
    def _get_input_files(self) -> List[str]:
        """Get list of input files to process"""
        if os.path.isfile(self.config.input_path):
            return [self.config.input_path]
        else:
            files = []
            for f in os.listdir(self.config.input_path):
                if f.endswith('.jsonl'):
                    files.append(os.path.join(self.config.input_path, f))
                    if len(files) >= self.config.max_files:
                        break
            return files

class NCLPreprocessor(BasePreprocessor):
    """Preprocessor for NCL data following the workflow"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        # Auto-generate field mappings
        self.field_mappings = self._generate_field_mappings()
        # Load and cache LLM config
        self.llm_config = self._load_llm_config()

    def _load_llm_config(self):
        try:
            with open("config/llm.yml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load LLM config: {e}")
            return {}
    
    def _generate_field_mappings(self) -> Dict:
        """Generate field mappings based on data analysis"""
        structure = self.analyze_structure()
        # Use ordered list to ensure longer (more specific) fields are matched first
        patterns = [
            ('title_en', ['論文名稱(外文)']),
            ('title', ['論文名稱']),
            ('school_en', ['學校名稱(外文)']),
            ('school', ['學校名稱']),
            ('dept_en', ['系所名稱(外文)']),
            ('dept', ['系所名稱']),
            ('abstract_en', ['摘要(外文)']),
            ('abstract', ['摘要']),
            ('authors_en', ['作者(外文)']),
            ('authors', ['作者']),
            ('advisors_en', ['指導教授(外文)']),
            ('advisors', ['指導教授']),
            ('id', ['uid']),
            ('year', ['畢業學年度']),
            ('category', ['學位類別']),
            ('link', ['博碩士論文網址'])
        ]
        mappings = {}
        for source_field in structure.keys():
            for target_field, keywords in patterns:
                for keyword in keywords:
                    if source_field == keyword:
                        mappings[source_field] = target_field
                        break
                if source_field in mappings:
                    break
        print("[DEBUG] Field mappings:", mappings)
        return mappings
    
    def extract_fields(self, raw_data: Dict) -> Dict:
        """Step 1: Extract and map fields from raw data"""
        extracted = {}
        
        for source_field, target_field in self.field_mappings.items():
            value = raw_data.get(source_field)
            if value is not None:
                extracted[target_field] = self._clean_value(value)
        
        # Generate ID if not present (using year + title as fallback)
        if 'id' not in extracted:
            year = extracted.get('year', 'unknown')
            title = extracted.get('title', 'unknown')
            extracted['id'] = f"{year}_{hash(title) % 10000}"
        
        return extracted
    
    def fill_content(self, extracted_fields: Dict) -> Dict:
        """Step 2: Fill content structure based on extracted fields"""
        content = {
            'id': extracted_fields.get('id'),
            'year': extracted_fields.get('year'),
            'category': extracted_fields.get('category'),
            'link': extracted_fields.get('link'),
            'chinese': {},
            'english': {},
            'keywords': []
        }
        
        # Fill Chinese content
        chinese_fields = {
            'title': extracted_fields.get('title'),
            'school': extracted_fields.get('school'),
            'dept': extracted_fields.get('dept'),
            'abstract': extracted_fields.get('abstract'),
            'authors': extracted_fields.get('authors'),
            'advisors': extracted_fields.get('advisors')
        }
        
        # Fill English content
        english_fields = {
            'title': extracted_fields.get('title_en'),
            'school': extracted_fields.get('school_en'),
            'dept': extracted_fields.get('dept_en'),
            'abstract': extracted_fields.get('abstract_en'),
            'authors': extracted_fields.get('authors_en'),
            'advisors': extracted_fields.get('advisors_en')
        }
        
        # Handle short abstracts - replace with title if abstract is too short
        chinese_abstract = chinese_fields.get('abstract')
        chinese_title = chinese_fields.get('title')
        if chinese_abstract and chinese_title and len(chinese_abstract.strip()) < 30:
            chinese_fields['abstract'] = chinese_title
        
        english_abstract = english_fields.get('abstract')
        english_title = english_fields.get('title')
        if english_abstract and english_title and len(english_abstract.strip()) < 30:
            english_fields['abstract'] = english_title
        
        # Add non-None values to content
        for field, value in chinese_fields.items():
            if value is not None:
                content['chinese'][field] = value
        
        for field, value in english_fields.items():
            if value is not None:
                content['english'][field] = value
        
        return content
    
    def preprocess_data(self, content: Dict) -> Optional[NCLDocument]:
        """Step 3: Preprocess content and create NCL_LLM_SummaryDocument"""
        # Validate required fields
        if not content['id']:
            return None
        # Create Info objects with proper list handling
        chinese_info = Info(
            title=content['chinese'].get('title'),
            school=content['chinese'].get('school'),
            dept=content['chinese'].get('dept'),
            abstract=content['chinese'].get('abstract'),
            authors=self._ensure_list(content['chinese'].get('authors')),
            advisors=self._ensure_list(content['chinese'].get('advisors'))
        )
        english_info = Info(
            title=content['english'].get('title'),
            school=content['english'].get('school'),
            dept=content['english'].get('dept'),
            abstract=content['english'].get('abstract'),
            authors=self._ensure_list(content['english'].get('authors')),
            advisors=self._ensure_list(content['english'].get('advisors'))
        )
        # Create and return NCLDocument
        return NCLDocument(
            id=content['id'],
            year=content['year'],
            category=content['category'],
            chinese=chinese_info,
            english=english_info,
            link=content['link'],
            keywords=content['keywords'],
        )
    
    def _clean_value(self, value: Any) -> Any:
        """Clean and validate a value"""
        if value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            
            # Convert year strings to integers
            if value.isdigit() and len(value) == 4:
                return int(value)
            
            return value
        
        if isinstance(value, (int, float)):
            return value
        
        return str(value) if value else None
    
    def _ensure_list(self, value: Any) -> List[str]:
        """Ensure value is a list of strings"""
        if value is None:
            return []
        
        if isinstance(value, str):
            # Split by newlines and clean
            return [line.strip() for line in value.splitlines() if line.strip()]
        
        if isinstance(value, list):
            return [str(item).strip() for item in value if item]
        
        return [str(value).strip()] if value else []
    

class LitSearchPreprocessor(BasePreprocessor):
    """Preprocessor for LitSearch data"""

    def extract_fields(self, raw_data: Dict) -> Dict:
        # Direct mapping, since LitSearch data is already normalized
        return {
            "corpusid": raw_data.get("corpusid"),
            "externalids": raw_data.get("externalids", {}),
            "title": raw_data.get("title"),
            "abstract": raw_data.get("abstract"),
            "authors": raw_data.get("authors"),
            "venue": raw_data.get("venue"),
            "year": raw_data.get("year"),
            "pdfurl": raw_data.get("pdfurl"),
        }

    def fill_content(self, extracted_fields: Dict) -> Dict:
        # For LitSearch, the extracted fields are already the content
        return extracted_fields

    def preprocess_data(self, content: Dict) -> Optional[LitSearchDocument]:
        # Validate required fields
        if not content.get("corpusid"):
            return None
        return LitSearchDocument(**content)
    


class NCL_LLM_SummaryPreprocessor(BasePreprocessor):
    """Preprocessor for NCL data following the workflow"""
    
    def __init__(self, config: PreprocessConfig):
        super().__init__(config)
        # Auto-generate field mappings
        self.field_mappings = self._generate_field_mappings()
        # Load and cache LLM config
        self.llm_config = self._load_llm_config()

    def _load_llm_config(self):
        try:
            with open("config/llm.yml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load LLM config: {e}")
            return {}
    
    def _generate_field_mappings(self) -> Dict:
        """Generate field mappings based on data analysis"""
        structure = self.analyze_structure()
        # Use ordered list to ensure longer (more specific) fields are matched first
        patterns = [
            ('title_en', ['論文名稱(外文)']),
            ('title', ['論文名稱']),
            ('school_en', ['學校名稱(外文)']),
            ('school', ['學校名稱']),
            ('dept_en', ['系所名稱(外文)']),
            ('dept', ['系所名稱']),
            ('abstract_en', ['摘要(外文)']),
            ('abstract', ['摘要']),
            ('authors_en', ['作者(外文)']),
            ('authors', ['作者']),
            ('advisors_en', ['指導教授(外文)']),
            ('advisors', ['指導教授']),
            ('id', ['uid']),
            ('year', ['畢業學年度']),
            ('category', ['學位類別']),
            ('link', ['博碩士論文網址'])
        ]
        mappings = {}
        for source_field in structure.keys():
            for target_field, keywords in patterns:
                for keyword in keywords:
                    if source_field == keyword:
                        mappings[source_field] = target_field
                        break
                if source_field in mappings:
                    break
        print("[DEBUG] Field mappings:", mappings)
        return mappings
    
    def extract_fields(self, raw_data: Dict) -> Dict:
        """Step 1: Extract and map fields from raw data"""
        extracted = {}
        
        for source_field, target_field in self.field_mappings.items():
            value = raw_data.get(source_field)
            if value is not None:
                extracted[target_field] = self._clean_value(value)
        
        # Generate ID if not present (using year + title as fallback)
        if 'id' not in extracted:
            year = extracted.get('year', 'unknown')
            title = extracted.get('title', 'unknown')
            extracted['id'] = f"{year}_{hash(title) % 10000}"
        
        return extracted
    
    def fill_content(self, extracted_fields: Dict) -> Dict:
        """Step 2: Fill content structure based on extracted fields"""
        content = {
            'id': extracted_fields.get('id'),
            'year': extracted_fields.get('year'),
            'category': extracted_fields.get('category'),
            'link': extracted_fields.get('link'),
            'chinese': {},
            'english': {},
            'keywords': []
        }
        
        # Fill Chinese content
        chinese_fields = {
            'title': extracted_fields.get('title'),
            'school': extracted_fields.get('school'),
            'dept': extracted_fields.get('dept'),
            'abstract': extracted_fields.get('abstract'),
            'authors': extracted_fields.get('authors'),
            'advisors': extracted_fields.get('advisors')
        }
        
        # Fill English content
        english_fields = {
            'title': extracted_fields.get('title_en'),
            'school': extracted_fields.get('school_en'),
            'dept': extracted_fields.get('dept_en'),
            'abstract': extracted_fields.get('abstract_en'),
            'authors': extracted_fields.get('authors_en'),
            'advisors': extracted_fields.get('advisors_en')
        }
        
        # Handle short abstracts - replace with title if abstract is too short
        chinese_abstract = chinese_fields.get('abstract')
        chinese_title = chinese_fields.get('title')
        if chinese_abstract and chinese_title and len(chinese_abstract.strip()) < 30:
            chinese_fields['abstract'] = chinese_title
        
        english_abstract = english_fields.get('abstract')
        english_title = english_fields.get('title')
        if english_abstract and english_title and len(english_abstract.strip()) < 30:
            english_fields['abstract'] = english_title
        
        # Add non-None values to content
        for field, value in chinese_fields.items():
            if value is not None:
                content['chinese'][field] = value
        
        for field, value in english_fields.items():
            if value is not None:
                content['english'][field] = value
        
        return content
    
    def preprocess_data(self, content: Dict) -> Optional[NCL_LLM_SummaryDocument]:
        """Step 3: Preprocess content and create NCL_LLM_SummaryDocument"""
        # Validate required fields
        if not content['id']:
            return None
        # Create Info objects with proper list handling
        chinese_info = Info(
            title=content['chinese'].get('title'),
            school=content['chinese'].get('school'),
            dept=content['chinese'].get('dept'),
            abstract=content['chinese'].get('abstract'),
            authors=self._ensure_list(content['chinese'].get('authors')),
            advisors=self._ensure_list(content['chinese'].get('advisors'))
        )
        english_info = Info(
            title=content['english'].get('title'),
            school=content['english'].get('school'),
            dept=content['english'].get('dept'),
            abstract=content['english'].get('abstract'),
            authors=self._ensure_list(content['english'].get('authors')),
            advisors=self._ensure_list(content['english'].get('advisors'))
        )
        # === LLM Summarization logic ===
        text_to_summarize = content['chinese'].get('abstract') or content['english'].get('abstract') or ""
        llm_questions = self._llm_summarize_text(text_to_summarize) if text_to_summarize else None
        keywords = self._llm_extract_keywords(text_to_summarize) if text_to_summarize else None
        # Create and return NCLDocument
        return NCL_LLM_SummaryDocument(
            id=content['id'],
            year=content['year'],
            category=content['category'],
            chinese=chinese_info,
            english=english_info,
            link=content['link'],
            keywords=keywords,
            llm_questions=llm_questions
        )
    
    def _clean_value(self, value: Any) -> Any:
        """Clean and validate a value"""
        if value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            
            # Convert year strings to integers
            if value.isdigit() and len(value) == 4:
                return int(value)
            
            return value
        
        if isinstance(value, (int, float)):
            return value
        
        return str(value) if value else None
    
    def _ensure_list(self, value: Any) -> List[str]:
        """Ensure value is a list of strings"""
        if value is None:
            return []
        
        if isinstance(value, str):
            # Split by newlines and clean
            return [line.strip() for line in value.splitlines() if line.strip()]
        
        if isinstance(value, list):
            return [str(item).strip() for item in value if item]
        
        return [str(value).strip()] if value else []
    
    def _llm_summarize_text(self, text: str) -> str:
        """
        Summarize text using OpenAI API (only current message, no history).
        """
        config = self.llm_config.get("openai", {})
        api_key = config.get("api_key")
        model = config.get("model", "gpt-4o-mini")
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 1500)

        if not api_key:
            print("[WARN] OpenAI API key not found in config. Returning original text.")
            return text

        client = openai.OpenAI(api_key=api_key)

        # System prompt (簡化版)
        system_prompt = (
            "請根據以下文章，生成一些完整、清楚、重點明確的中文問句，"
            "重點放在文章核心內容與詮釋，以問句列表呈現，不要超過5種問句。"
            "請勿出現過於簡單的問句，例如：「本論文的研究動機與目的為何？」"
            "問句列表以逗號','分隔，不要有任何其他文字。"
            "請將問句內容放在 <QUESTIONS> 和 </QUESTIONS> 標記之間。"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content.strip()

            # 自動補齊結尾標籤（避免被截斷）
            if "<QUESTIONS>" in content and "</QUESTIONS>" not in content:
                content += "</QUESTIONS>"

            # 擷取 QUESTIONS 內容
            match = re.search(r"<QUESTIONS>(.*?)</QUESTIONS>", content, re.DOTALL)
            if match:
                summary = match.group(1).strip()
            else:
                # 標籤缺失也回傳純文字
                summary = content.replace("<QUESTIONS>", "").replace("</QUESTIONS>", "").strip()

            # 將問句列表轉換為列表
            questions = [question.strip() for question in summary.split(',')]

            return questions

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            return text
    
    def _llm_extract_keywords(self, text: str) -> str:
        """
        Extract keywords from text using OpenAI API (only current message, no history).
        """
        config = self.llm_config.get("openai", {})
        api_key = config.get("api_key")
        model = config.get("model", "gpt-4o")
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 1500)

        if not api_key:
            print("[WARN] OpenAI API key not found in config. Returning original text.")
            return text

        client = openai.OpenAI(api_key=api_key)

        # System prompt (簡化版)
        system_prompt = (
            "請根據以下文章，生成一些完整、清楚、重點明確的中文關鍵字，"
            "重點放在文章核心內容與詮釋，以關鍵字列表呈現，不要超過10種關鍵字。"
            "關鍵字列表以逗號','分隔，不要有任何其他文字。"
            "請將關鍵字內容放在 <KEYWORDS> 和 </KEYWORDS> 標記之間。"
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            content = response.choices[0].message.content.strip()

            # 自動補齊結尾標籤（避免被截斷）
            if "<KEYWORDS>" in content and "</KEYWORDS>" not in content:
                content += "</KEYWORDS>"

            # 擷取 KEYWORDS 內容
            match = re.search(r"<KEYWORDS>(.*?)</KEYWORDS>", content, re.DOTALL)
            if match:
                summary = match.group(1).strip()
            else:
                # 標籤缺失也回傳純文字
                summary = content.replace("<KEYWORDS>", "").replace("</KEYWORDS>", "").strip()
            
            # 將關鍵字列表轉換為列表
            keywords = [keyword.strip() for keyword in summary.split(',')]

            return keywords

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed: {e}")
            return text

class PreprocessorFactory:
    """Factory for creating preprocessors"""
    
    @classmethod
    def create(cls, dataset_type: str, config: PreprocessConfig) -> BasePreprocessor:
        """Create a preprocessor for the specified dataset type"""
        if dataset_type == "ncl":
            return NCLPreprocessor(config)
        elif dataset_type == "litsearch":
            return LitSearchPreprocessor(config)
        elif dataset_type == "ncl_llm_summary":
            return NCL_LLM_SummaryPreprocessor(config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")