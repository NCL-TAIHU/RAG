import unittest
import tempfile
import os
import shutil
import json
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.sparse import csr_array
import types
from sentence_transformers import SentenceTransformer

# Import core components
from src.core.document import Document, NCLDocument, LitSearchDocument, Info, Field, FieldType
from src.core.filter import Filter, NCLFilter, LitSearchFilter
from src.core.chunker import BaseChunker, LengthChunker, SentenceChunker, ChunkerMetaData
from src.core.embedder import DenseEmbedder, SparseEmbedder
from src.core.vector_store import FileBackedDenseVS, FileBackedSparseVS, VSMetadata
from src.core.vector_manager import VectorManager
from src.core.search_engine import MilvusSearchEngine, HybridMilvusSearchEngine
from src.core.library import InMemoryLibrary
from src.core.manager import Manager
from src.core.reranker import IdentityReranker
from src.core.router import SimpleRouter
from src.core.data import DataLoader
from src.run.app import SearchApp
from src.run.factory import AppFactory

# Import for datetime
from datetime import datetime


class NoChunkChunker(BaseChunker):
    """
    A chunker that doesn't actually chunk - returns the entire document as one chunk.
    Used for testing retrieval without chunking.
    """
    def chunk(self, docs: List[str]) -> List[List[str]]:
        return [[doc] for doc in docs]
    
    def metadata(self) -> ChunkerMetaData:
        return ChunkerMetaData(chunker_type="no_chunk", params={})


class RealDenseEmbedder(DenseEmbedder):
    _model_cache = {}

    def __init__(self, model_name='BAAI/bge-m3'):
        if model_name not in RealDenseEmbedder._model_cache:
            RealDenseEmbedder._model_cache[model_name] = SentenceTransformer(model_name)
        self.model = RealDenseEmbedder._model_cache[model_name]
        self._name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def name(self) -> str:
        return self._name
    def dimension(self) -> int:
        return self._dimension
    def get_dim(self):
        return self._dimension


class MockSparseEmbedder(SparseEmbedder):
    """
    Mock sparse embedder for testing that returns deterministic sparse embeddings.
    """
    def __init__(self, dimension: int = 1000):
        self._dimension = dimension
        self._name = "mock-sparse-embedder"
    
    def embed(self, texts: List[str]) -> csr_array:
        """Generate deterministic sparse embeddings based on text content."""
        embeddings = []
        for text in texts:
            # Create a deterministic sparse embedding based on text hash
            seed = hash(text) % 10000
            np.random.seed(seed)
            # Generate sparse vector with ~10% non-zero elements
            dense_vector = np.random.rand(self._dimension)
            # Make it sparse by zeroing out 90% of elements
            mask = np.random.rand(self._dimension) > 0.9
            dense_vector = dense_vector * mask
            embeddings.append(dense_vector)
        
        return csr_array(embeddings)
    
    def name(self) -> str:
        return self._name

    def get_dim(self):
        return self._dimension


class MockDataLoader(DataLoader):
    """
    Mock data loader that generates fake documents for testing.
    """
    def __init__(self, num_docs: int = 10):
        self.num_docs = num_docs
        self.documents = []
    
    def load(self):
        """Generate fake documents and yield them in batches."""
        # Generate fake NCL documents
        for i in range(self.num_docs):
            doc = NCLDocument(
                id=f"test_doc_{i}",
                year=2020 + (i % 5),  # Years 2020-2024
                category="碩士" if i % 2 == 0 else "博士",
                chinese=Info(
                    title=f"測試論文標題 {i}",
                    school=f"測試大學 {i % 3}",
                    dept=f"測試系所 {i % 4}",
                    abstract=f"這是第{i}篇論文的摘要。這篇論文討論了機器學習和自然語言處理的相關主題。我們提出了一種新的方法來解決文本分類問題。實驗結果表明，我們的方法在準確率和效率方面都優於現有的方法。這個研究對於推動人工智能領域的發展具有重要意義。",
                    authors=[f"作者{i}"],
                    advisors=[f"指導教授{i}"]
                ),
                english=Info(
                    title=f"Test Thesis Title {i}",
                    school=f"Test University {i % 3}",
                    dept=f"Test Department {i % 4}",
                    abstract=f"This is the abstract of thesis {i}. This thesis discusses topics related to machine learning and natural language processing. We propose a new method to solve text classification problems. Experimental results show that our method outperforms existing methods in both accuracy and efficiency. This research is significant for advancing the field of artificial intelligence.",
                    authors=[f"Author{i}"],
                    advisors=[f"Advisor{i}"]
                ),
                link=f"https://example.com/thesis_{i}",
                keywords=[f"機器學習", f"自然語言處理", f"文本分類", f"人工智能"]
            )
            self.documents.append(doc)
        
        # Yield documents in batches
        batch_size = 5
        for i in range(0, len(self.documents), batch_size):
            yield self.documents[i:i + batch_size]


class RetrievalComparisonTest(unittest.TestCase):
    """
    Comprehensive test that compares retrieval results before and after chunking.
    """
    # --- EDIT YOUR DOCUMENTS HERE ---
    DOCUMENTS_DATA = [
        {
            "id": "doc1",
            "year": 2024,
            "category": "碩士",
            "chinese": {
                "title": "基於大型語言模型之對話式推薦系統：以中藥方推薦為例",
                "school": "中央大學",
                "dept": "資訊工程學系",
                "abstract": "傳統的推薦系統主要依賴於分析數據和機器學習算法，並由系統單方面向使用者推播。對話式推薦系統則可以直接接受來自使用者主動提供的資訊，而系統也能透過文字對項目進行推薦，給予最直接的幫助。 在對話式推薦系統（Conversational Recommender System, CRS）中使用大型語言模型（Large Language Model, LLM）能獲得許多傳統模型無法擁有的優勢。首先，LLM的系統不需要經過訓練即可展現出色的性能，能夠解決冷啟動問題。其次，LLM的泛用性及可擴展性極高，能適應或導入到各種應用場景。 大多數以往的研究偏向於從大量的物件中推薦出一個最相關的內容，這較適合使用檢索增強生成（Retrieval Augmented Generation, RAG）技術；我們的研究著重於從少量的物件中推薦一個最適合的內容，並強調應依據病因來推薦藥方。 本研究首次嘗試以失眠患者的中藥方推薦作為任務目標，使用繁體中文能力出色的gemini-1.5-flash模擬並生成病患與中醫師之間的對話。我們提出Hint Module來導入生理量測及中醫問診技巧，透過偵測特定字串來觸發系統機制，給予LLM額外的提示訊息藉此影響它的輸出結果。 我們的實驗結果顯示，此方法可生成寫實的自述，能被視作為良好的對話範例，並在十種中藥方的推薦任務中，可得到八成以上的準確率及Macro-F1成績。其中Hint Module能顯著地改善多輪對話後的成績表現 (p-value < 0.01)。 最後我們也以中醫學的各項觀點進行分析，透過視覺化的圖表呈現出各個藥方之間分布上的關聯性，以得到更全面及清晰的瞭解。 實驗結果展示了以LLM打造的中藥方推薦系統擁有傑出的基礎能力，並擁有絕佳的可擴展性。",
                "authors": ["張廷睿"],
                "advisors": ["蔡宗翰"]
            },
            "english": {
                "title": "Conversational Recommender System based on Large Language Model: A Case Study in Traditional Chinese Medicine Recommendation",
                "school": "National Central University",
                "dept": "Department of Computer Science and Information Engineering",
                "abstract": "Traditional recommender systems primarily rely on data analysis and machine learning algorithms, with recommendations being pushed to users unilaterally by the system. In contrast, conversational recommender systems can directly accept information actively provided by users, and the system can recommend items through text, offering the most direct assistance. Using a Large Language Model (LLM) in a Conversational Recommender System (CRS) offers many advantages that traditional models do not have. First, an LLM system does not require training to perform well, which solves the cold-start problem. Second, LLMs possess high versatility and scalability, allowing them to adapt or be integrated into various application scenarios. Most previous studies focused on recommending the most relevant content from a large set of items, which is more suitable for Retrieval Augmented Generation (RAG) techniques. Our research, however, focuses on recommending the most appropriate content from a small set of items, emphasizing the recommendation of prescriptions based on the cause of illness. Our study is the first to attempt using Traditional Chinese Medicine (TCM) prescription recommendations for insomnia patients as a task goal, utilizing gemini-1.5-flash, which excels in Traditional Chinese, to simulate and generate conversations between patients and TCM doctors. We propose using a Hint Module to incorporate physiological measurements and consultation techniques, triggering the system to provide LLM with additional prompt messages by detecting specific strings, thereby influencing the LLM’s output. Our experimental results show that this method can generate realistic patient statements that can be regarded as good examples, achieving over 80% accuracy and Macro-F1 score in recommending ten classes of prescriptions. The Hint Module can significantly (p-value < 0.01) improve the performance of the multiple rounds of dialogue. Lastly, we conducted an analysis based on various perspectives of TCM, we used visualized charts to present the distributional relationships among different prescriptions, aiming for a more comprehensive and clearer understanding. The experimental results show that the TCM prescription recommender system built with LLM has excellent basic capabilities and scalability.",
                "authors": ["Ting-Jui Chang"],
                "advisors": ["Tzong-Han Tsai"]
            },
            "link": "https://example.com/medimg",
            "keywords": ["深度學習", "醫學影像", "卷積神經網路"]
        },
        {
            "id": "doc2",
            "year": 2024,
            "category": "碩士",
            "chinese": {
                "title": "影視教材在國中台灣史的素養教學運用─以《灣生回家》為例",
                "school": "國立臺灣師範大學",
                "dept": "歷史學系歷史教學碩士在職專班",
                "abstract": "日治時期的台灣，除了台灣人之外，也有生活在同片土地上的日本管理者及日本移民。「灣生」議題在2015年前以「遣返者」一詞較為熟知，直到紀錄片《灣生回家》的出現，才讓灣生一詞廣為人知。本研究於國中素養課程中，使用影視教材來做運用，導演黃銘正於2017年因書籍作者造假身分，澄清書籍與紀錄片之間應有所區別，勿忽略還有這樣的灣生真實存在。關切紀錄片中的灣生心態，經歷過的童年與被遣返後的區別對待，促成灣生的追尋之旅。本研究集中於當代的時空背景及灣生的身分認同感，再分析導演的理念及呈現手法，最後透過本紀錄片設計課程教案，研究結果顯示學生對於這樣的影視教材的確有提升對議題的認識，進而化為行動，也展現他們的同理心，符合108課綱要求的素養精神。",
                "authors": ["廖翊伶"],
                "advisors": ["陳登武"]
            },
            "english": {
                "title": "Employing audiovisual materials in Taiwanese history teaching in Junior High Schools：A case study of Wansei Back Home",
                "school": "National Taiwan Normal University",
                "dept": "Department of History",
                "abstract": """
                            During the Japanese colonial period in Taiwan, alongside the Taiwanese people, there were also Japanese administrators and immigrants living on the same land. Before 2015, the issue of " Wansei " (Taiwanese born to Japanese parents) was more commonly known by the term "repatriates." It wasn't until the documentary " Wansei Back Home " that the term " Wansei " gained widespread recognition. This study utilizes audiovisual materials in the junior high school curriculum. In 2017, director Huang Mingzheng clarified the distinction between the book and the documentary due to the author’s falsified identity, emphasizing that the real existence of Wansei should not be overlooked. The documentary sheds light on the mindset of the Wansei, exploring the differences in their childhood experiences and treatment after repatriation, which contributes to their journey of self-discovery. This research focuses on the contemporary historical context and the identity of the Wansei, analyzing the director's philosophy and presentation techniques. Finally, through the design of lesson plans based on this documentary, the research results show that students indeed gain a better understanding of the issue through such audiovisual materials, translating this knowledge into action and demonstrating empathy, which aligns with the competencies required by the 108 Curriculum Guidelines.
                            """,
                "authors": ["Yi-Ling Liao"],
                "advisors": ["Chen, Deng-Wu"]
            },
            "link": "https://example.com/nlp",
            "keywords": ["自然語言處理", "語意分析"]
        },
        {
            "id": "doc3",
            "year": 2019,
            "category": "碩士",
            "chinese": {
                "title": "固態硬碟內位址映射表之錯誤模擬、檢測、及修復",
                "school": "國立清華大學",
                "dept": "電機工程學系",
                "abstract": "摘要隱藏中",
                "authors": ["李柏霖"],
                "advisors": ["呂仁碩"]
            },
            "english": {
                "title": "Error Simulation, Detection, and Repair of Address Mapping Tables in Solid-State Drives",
                "school": "National Tsing Hua University",
                "dept": "Department of Electrical Engineering",
                "abstract": """
                            The abstract is hidden.
                            """,
                "authors": ["Bolin Li"],
                "advisors": ["Ren-Shuo Lu"]
            },
            "link": "https://example.com/nlp",
            "keywords": ["自然語言處理", "語意分析"]
        }
        # Add more documents as needed...
    ]

    def setUp(self):
        """Set up test environment with temporary directories and user-editable documents."""
        # Patch model_config for real and mock embedders
        patcher = patch("src.core.vector_manager.model_config", {
            "sentence-transformers/all-MiniLM-L6-v2": {"alias": "sentence-transformers/all-MiniLM-L6-v2", "type": "dense"},
            "BAAI/bge-m3": {"alias": "BAAI/bge-m3", "type": "sparse"},
            "mock-sparse-embedder": {"alias": "mock-sparse-embedder", "type": "sparse"}
        })
        self.addCleanup(patcher.stop)
        patcher.start()
        
        # Create temporary directories for vector stores
        self.temp_dir = tempfile.mkdtemp()
        self.no_chunk_dir = os.path.join(self.temp_dir, "no_chunk")
        self.chunked_dir = os.path.join(self.temp_dir, "chunked")
        os.makedirs(self.no_chunk_dir, exist_ok=True)
        os.makedirs(self.chunked_dir, exist_ok=True)
        
        # Test parameters
        self.test_queries = [
            "深度學習",
            "自然語言處理",
            "是否已有研究探討將大型語言模型（Large Language Models, LLMs）應用於對話式推薦系統（Conversational Recommender Systems, CRS），以解決冷啟動問題，並實作在中醫藥方推薦（特別是針對失眠患者）上，進行多輪問答、引導式對話以及結合中醫診療技巧來提升推薦準確率？"
        ]
        
        # --- Convert DOCUMENTS_DATA to NCLDocument objects ---
        self.documents = []
        for doc in self.DOCUMENTS_DATA:
            ncl_doc = NCLDocument(
                id=doc["id"],
                year=doc.get("year"),
                category=doc.get("category"),
                chinese=Info(**doc["chinese"]),
                english=Info(**doc["english"]),
                link=doc.get("link"),
                keywords=doc.get("keywords", [])
            )
            self.documents.append(ncl_doc)

        self.dense_embedder = RealDenseEmbedder()

    def create_vector_manager(self, chunker: BaseChunker, vs_dir: str, embedding_type: str = "dense") -> VectorManager:
        """Create a vector manager with the specified chunker and embedding type."""
        if embedding_type == "dense":
            embedder = self.dense_embedder
            model_name = embedder.name()
            embedding_type_str = "dense"
        else:
            embedder = MockSparseEmbedder()
            model_name = embedder.name()
            embedding_type_str = "sparse"

        metadata = VSMetadata(
            embedding_type=embedding_type_str,
            dataset="test_dataset",
            channel="abstract_chinese",
            chunker_meta=chunker.metadata(),
            model=model_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        if embedding_type == "dense":
            vector_store = FileBackedDenseVS(vs_dir, metadata)
        else:
            vector_store = FileBackedSparseVS(vs_dir, metadata)
        return VectorManager(vector_store, embedder, chunker, "test_dataset", "abstract_chinese")
    
    def create_search_engine(self, vector_manager: VectorManager, embedding_type: str = "dense") -> MilvusSearchEngine:
        """Create a mock search engine that doesn't require Milvus."""
        engine = MilvusSearchEngine(
            vector_type=embedding_type,
            vector_manager=vector_manager,
            document_cls=NCLDocument,
            filter_cls=NCLFilter,
            force_rebuild=True
        )
        # Mock the operator attribute directly
        mock_operator = MagicMock()
        def mock_search(data, anns_field, param, limit, expr, output_fields):
            doc_ids = [f"test_doc_{i}" for i in range(min(limit, self.num_docs))]
            return [(MagicMock(fields={"pk": doc_id}) for doc_id in doc_ids)]
        mock_operator.search = mock_search
        engine.operator = mock_operator

        # Mock the collection attribute
        mock_collection = MagicMock()
        mock_collection.load = MagicMock()
        mock_collection.query = MagicMock(return_value=[])
        engine.collection = mock_collection

        # Mock the embed_query method
        def mock_embed_query(query: str):
            return vector_manager.get_raw_embedding([query])[0]
        engine.embed_query = mock_embed_query

        # Patch insert to handle sparse embeddings correctly
        def patched_insert(self_engine, documents):
            existing_pks = set()
            if documents:
                keys = [doc.key() for doc in documents]
                existing_pks = set()
            new_docs = [doc for doc in documents if doc.key() not in existing_pks]
            if not new_docs:
                return

            dense_embeddings = self_engine.vm.get_doc_embeddings(new_docs)
            sparse_embeddings = self_engine.vm.get_doc_embeddings(new_docs)

            insert_dict = {
                "pk": [],
                "dense_vector": [],
                "sparse_vector": [],
            }

            for doc in new_docs:
                doc_id = doc.key()
                dense_chunks = dense_embeddings[doc_id]
                sparse_chunks = sparse_embeddings[doc_id]
                # Use .shape[0] for csr_array, len() for list
                if isinstance(dense_chunks, list):
                    n_chunks = len(dense_chunks)
                else:
                    n_chunks = dense_chunks.shape[0]
                if isinstance(sparse_chunks, list):
                    n_chunks_sparse = len(sparse_chunks)
                else:
                    n_chunks_sparse = sparse_chunks.shape[0]
                assert n_chunks == n_chunks_sparse
                for i in range(n_chunks):
                    insert_dict["pk"].append(f"{doc_id}-{i}")
                    insert_dict["dense_vector"].append(dense_chunks[i] if isinstance(dense_chunks, list) else dense_chunks[i].toarray().flatten())
                    insert_dict["sparse_vector"].append(sparse_chunks[i] if isinstance(sparse_chunks, list) else sparse_chunks[i])
            # skip metadata for test
            # No actual DB insert needed in test
            return

        engine.insert = types.MethodType(patched_insert, engine)

        return engine
    
    def test_retrieval_comparison_no_chunking_vs_chunking(self):
        """Test retrieval results comparison between no chunking and chunking."""
        print("\n" + "="*80)
        print("TESTING RETRIEVAL COMPARISON: NO CHUNKING vs CHUNKING")
        print("="*80)
        
        # Create chunkers
        no_chunk_chunker = NoChunkChunker()
        sentence_chunker = SentenceChunker(language="zh")
        length_chunker = LengthChunker(chunk_length=30)
        
        # Test different chunking strategies
        chunking_strategies = [
            ("no_chunk", no_chunk_chunker),
            ("sentence", sentence_chunker),
            ("length_30", length_chunker)
        ]
        
        results_summary = {}
        
        for strategy_name, chunker in chunking_strategies:
            print(f"\n--- Testing {strategy_name} strategy ---")
            
            # Create vector managers for both dense and sparse
            dense_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/{strategy_name}_dense", "dense")
            sparse_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/{strategy_name}_sparse", "sparse")
            
            # Insert documents
            dense_vm.vector_store.insert(dense_vm.get_doc_embeddings(self.documents))
            sparse_vm.vector_store.insert(sparse_vm.get_doc_embeddings(self.documents))
            
            # Save vector stores
            dense_vm.vector_store.save()
            sparse_vm.vector_store.save()
            
            # Test retrieval for each query
            strategy_results = {}
            
            for query in self.test_queries:
                print(f"  Query: '{query}'")
                
                # Get embeddings for the query
                dense_query_embedding = dense_vm.get_raw_embedding([query])[0]
                sparse_query_embedding = sparse_vm.get_raw_embedding([query])[0]
                
                # Calculate similarities with all documents
                dense_similarities = []
                sparse_similarities = []
                
                for doc in self.documents:
                    doc_embeddings = dense_vm.get_doc_embeddings([doc])
                    doc_dense_embeddings = doc_embeddings[doc.key()]
                    doc_sparse_embeddings = sparse_vm.get_doc_embeddings([doc])[doc.key()]
                    
                    # Calculate dense similarity (cosine similarity)
                    if isinstance(doc_dense_embeddings[0], list):
                        doc_dense_vec = np.array(doc_dense_embeddings[0])
                        query_dense_vec = np.array(dense_query_embedding)
                        dense_sim = np.dot(doc_dense_vec, query_dense_vec) / (np.linalg.norm(doc_dense_vec) * np.linalg.norm(query_dense_vec))
                        dense_similarities.append((doc.key(), dense_sim))
                    
                    # Calculate sparse similarity (cosine similarity)
                    if isinstance(doc_sparse_embeddings, csr_array):
                        doc_sparse_vec = doc_sparse_embeddings[0].toarray().flatten()
                        query_sparse_vec = sparse_query_embedding.toarray().flatten()
                        sparse_sim = np.dot(doc_sparse_vec, query_sparse_vec) / (np.linalg.norm(doc_sparse_vec) * np.linalg.norm(query_sparse_vec))
                        sparse_similarities.append((doc.key(), sparse_sim))
                
                # Sort by similarity and get top 5
                dense_similarities.sort(key=lambda x: x[1], reverse=True)
                sparse_similarities.sort(key=lambda x: x[1], reverse=True)
                
                top_dense = [doc_id for doc_id, _ in dense_similarities[:5]]
                top_sparse = [doc_id for doc_id, _ in sparse_similarities[:5]]
                
                strategy_results[query] = {
                    "dense_top5": top_dense,
                    "sparse_top5": top_sparse,
                    "dense_similarities": dict(dense_similarities[:5]),
                    "sparse_similarities": dict(sparse_similarities[:5])
                }
                
                print(f"    Dense top 5: {top_dense}")
                print(f"    Sparse top 5: {top_sparse}")
            
            results_summary[strategy_name] = strategy_results
        
        # Analyze results
        self.analyze_retrieval_results(results_summary)
    
    def analyze_retrieval_results(self, results_summary: Dict[str, Dict]):
        """For each query and chunking strategy, print only the top-5 document IDs and their retrieval scores for dense and sparse retrieval."""
        print("\n" + "="*80)
        print("RETRIEVAL RESULTS ANALYSIS")
        print("="*80)
        
        strategies = list(results_summary.keys())
        queries = list(results_summary[strategies[0]].keys())
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            for strategy in strategies:
                dense_top5 = results_summary[strategy][query]["dense_top5"]
                dense_sims = results_summary[strategy][query]["dense_similarities"]
                print(f"  Dense Retrieval ({strategy}):")
                for doc_id in dense_top5:
                    score = dense_sims.get(doc_id, 0.0)
                    print(f"    {doc_id}: {score:.3f}")
                sparse_top5 = results_summary[strategy][query]["sparse_top5"]
                sparse_sims = results_summary[strategy][query]["sparse_similarities"]
                print(f"  Sparse Retrieval ({strategy}):")
                for doc_id in sparse_top5:
                    score = sparse_sims.get(doc_id, 0.0)
                    print(f"    {doc_id}: {score:.3f}")
    
    def test_chunking_impact_on_embedding_quality(self):
        """Test how chunking affects embedding quality and retrieval performance."""
        print("\n" + "="*80)
        print("TESTING CHUNKING IMPACT ON EMBEDDING QUALITY")
        print("="*80)
        
        # Create different chunkers
        no_chunk_chunker = NoChunkChunker()
        sentence_chunker = SentenceChunker(language="zh")
        length_chunker_30 = LengthChunker(chunk_length=30)
        length_chunker_100 = LengthChunker(chunk_length=100)
        
        chunkers = [
            ("no_chunk", no_chunk_chunker),
            ("sentence", sentence_chunker),
            ("length_30", length_chunker_30),
            ("length_100", length_chunker_100)
        ]
        
        # Test with a specific document
        test_doc = self.documents[0]
        test_text = test_doc.chinese.abstract
        
        print(f"Test document abstract: {test_text[:100]}...")
        
        for chunker_name, chunker in chunkers:
            print(f"\n--- {chunker_name} chunker ---")
            
            # Chunk the text
            chunks = chunker.chunk([test_text])[0]
            print(f"  Number of chunks: {len(chunks)}")
            
            # Show chunk details
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk[:50]}... (length: {len(chunk)})")
            
            # Create vector manager and get embeddings
            dense_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/quality_test_{chunker_name}", "dense")
            
            # Get embeddings for the document
            doc_embeddings = dense_vm.get_doc_embeddings([test_doc])
            embeddings = doc_embeddings[test_doc.key()]
            
            print(f"  Number of embeddings: {len(embeddings)}")
            print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            
            # Calculate embedding statistics
            if len(embeddings) > 1:
                # Calculate average pairwise cosine similarity between chunks
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        vec1 = np.array(embeddings[i])
                        vec2 = np.array(embeddings[j])
                        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                std_similarity = np.std(similarities)
                print(f"  Average inter-chunk similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    def test_end_to_end_retrieval_simulation(self):
        """Simulate end-to-end retrieval process with different chunking strategies."""
        print("\n" + "="*80)
        print("END-TO-END RETRIEVAL SIMULATION")
        print("="*80)
        
        # Create library and search engines
        library = InMemoryLibrary()
        library.insert(self.documents)
        
        # Test different chunking strategies
        chunking_strategies = [
            ("no_chunk", NoChunkChunker()),
            ("sentence", SentenceChunker(language="zh")),
            ("length_30", LengthChunker(chunk_length=30))
        ]
        
        for strategy_name, chunker in chunking_strategies:
            print(f"\n--- Testing {strategy_name} strategy ---")
            
            # Create vector managers
            dense_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/e2e_{strategy_name}_dense", "dense")
            sparse_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/e2e_{strategy_name}_sparse", "sparse")
            
            # Create search engines (mocked)
            dense_engine = self.create_search_engine(dense_vm, "dense")
            sparse_engine = self.create_search_engine(sparse_vm, "sparse")
            
            # Create manager
            manager = Manager(
                library=library,
                search_engines=[dense_engine, sparse_engine],
                reranker=IdentityReranker(),
                router_name="simple"
            )
            
            # Insert documents into search engines
            dense_engine.insert(self.documents)
            sparse_engine.insert(self.documents)
            
            # Test retrieval
            for query in self.test_queries[:3]:  # Test first 3 queries
                print(f"  Query: '{query}'")
                
                # Test dense search
                try:
                    dense_results = dense_engine.search(query, NCLFilter.EMPTY, limit=5)
                    print(f"    Dense results: {dense_results}")
                except Exception as e:
                    print(f"    Dense search error: {e}")
                
                # Test sparse search
                try:
                    sparse_results = sparse_engine.search(query, NCLFilter.EMPTY, limit=5)
                    print(f"    Sparse results: {sparse_results}")
                except Exception as e:
                    print(f"    Sparse search error: {e}")
    
    def test_chunking_metadata_consistency(self):
        """Test that chunking metadata is properly maintained throughout the pipeline."""
        print("\n" + "="*80)
        print("TESTING CHUNKING METADATA CONSISTENCY")
        print("="*80)
        
        chunkers = [
            ("no_chunk", NoChunkChunker()),
            ("sentence", SentenceChunker(language="zh")),
            ("length_30", LengthChunker(chunk_length=30))
        ]
        
        for chunker_name, chunker in chunkers:
            print(f"\n--- {chunker_name} chunker ---")
            
            # Test chunker metadata
            metadata = chunker.metadata()
            print(f"  Chunker type: {metadata.chunker_type}")
            print(f"  Chunker params: {metadata.params}")
            
            # Create vector manager and check metadata consistency
            dense_vm = self.create_vector_manager(chunker, f"{self.temp_dir}/metadata_{chunker_name}", "dense")
            vs_metadata = dense_vm.get_vs_metadata()
            
            print(f"  VS chunker type: {vs_metadata.chunker_meta.chunker_type}")
            print(f"  VS chunker params: {vs_metadata.chunker_meta.params}")
            
            # Verify metadata consistency
            assert metadata.chunker_type == vs_metadata.chunker_meta.chunker_type, \
                f"Chunker type mismatch for {chunker_name}"
            assert metadata.params == vs_metadata.chunker_meta.params, \
                f"Chunker params mismatch for {chunker_name}"
            
            print(f"  ✅ Metadata consistency verified")


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 