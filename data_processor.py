import os
from typing import Tuple, Dict, List, Any
from loguru import logger
from milvus_model.hybrid import BGEM3EmbeddingFunction
from config import Config

class DataProcessor:
    """資料處理類別，負責資料的載入和處理"""
    
    def __init__(self):
        """初始化資料處理器"""
        self.embedding_function = None
    
    def initialize_embedding_function(self) -> BGEM3EmbeddingFunction:
        """初始化 BGE-M3 嵌入函數"""
        try:
            self.embedding_function = BGEM3EmbeddingFunction(device="cuda")
            logger.info("Embedding function initialized successfully")
            return self.embedding_function
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {str(e)}")
            raise
    
    def load_and_process_data(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """載入並處理資料"""
        try:
            metadata = {
                "id": [],
                "data": [],
                "content": [],
            }
            abstract = []
            
            logger.info("Starting to load and process data")
            
            for i in range(1, Config.NUM_FILES + 1):
                filename = f"{i}.txt"
                file1_path = os.path.join(Config.ABSTRACT_PATH, filename)
                file2_path = os.path.join(Config.DATA_PATH, filename)
                file3_path = os.path.join(Config.CONTENT_PATH, filename)
                
                try:
                    with open(file1_path, "r", encoding="utf-8") as f1, \
                         open(file2_path, "r", encoding="utf-8") as f2, \
                         open(file3_path, "r", encoding="utf-8") as f3:
                        content1 = f1.read()
                        content2 = f2.read()
                        content3 = f3.read()
                        metadata['id'].append(str(i))
                        metadata['data'].append(content2)
                        metadata['content'].append(content3)
                        abstract.append(content1)
                except FileNotFoundError as e:
                    logger.warning(f"File not found: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing file {filename}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(abstract)} documents")
            return metadata, abstract
            
        except Exception as e:
            logger.error(f"Error in load_and_process_data: {str(e)}")
            raise
    
    def generate_embeddings(self, abstract: List[str], ef: BGEM3EmbeddingFunction) -> Tuple[Dict[str, Any], int]:
        """生成嵌入向量"""
        try:
            logger.info("Starting to generate embeddings")
            docs_embeddings = ef(abstract)
            dense_dim = ef.dim["dense"]
            logger.info(f"Successfully generated embeddings with dimension {dense_dim}")
            return docs_embeddings, dense_dim
        except Exception as e:
            logger.error(f"Error in generate_embeddings: {str(e)}")
            raise 