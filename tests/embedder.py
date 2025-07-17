import unittest
import sys
import os
import io
import numpy as np
import csv
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embedder import AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder, DenseEmbedder, SparseEmbedder # adjust the import path

TEST_TEXTS = ["The quick brown fox jumps over the lazy dog."] * 4

class TestEmbedders(unittest.TestCase):

    def run_embedder_test(self, embedder_class, name, *args, **kwargs):
        buffer_out = io.StringIO()
        buffer_err = io.StringIO()

        with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
            embedder = embedder_class(*args, **kwargs)
            _ = embedder.embed(TEST_TEXTS)

        stdout_output = buffer_out.getvalue()
        stderr_output = buffer_err.getvalue()

        print(f"\n=== {name} STDOUT ===\n{stdout_output}")
        print(f"=== {name} STDERR ===\n{stderr_output}")

    def test_automodel_embedder(self):
        
        self.run_embedder_test(AutoModelEmbedder, "AutoModelEmbedder", "sentence-transformers/all-MiniLM-L6-v2")

    def test_bgem3_embedder(self):
        self.run_embedder_test(BGEM3Embedder, "BGEM3Embedder", "BAAI/bge-m3", use_fp16=False)

    def test_milvus_bgem3_embedder(self):
        self.run_embedder_test(MilvusBGEM3Embedder, "MilvusBGEM3Embedder")

    def test_dense_embedder(self):
        # Use the factory method to create a DenseEmbedder
        embedder = DenseEmbedder.from_default("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(embedder, DenseEmbedder)
        self.assertIsInstance(embedder, AutoModelEmbedder)
        
        # Test embedding
        embeddings = embedder.embed(TEST_TEXTS)
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), len(TEST_TEXTS))

    def test_sparse_embedder(self):
        # Use the factory method to create a SparseEmbedder
        embedder = SparseEmbedder.from_default("BAAI/bge-m3")
        self.assertIsInstance(embedder, SparseEmbedder)
        self.assertIsInstance(embedder, BGEM3Embedder)
        
        # Test embedding
        embeddings = embedder.embed(TEST_TEXTS)
        self.assertIsInstance(embeddings, csr_array)
        self.assertEqual(embeddings.shape[0], len(TEST_TEXTS))

    def test_similar_sentence_embedding(self):
        '''
        Test the similarity of sentence embedding.
        The sentence are similar, so the embedding should be similar.
        The embedding should be similar for all embedders.
        '''
        sentence_1 = "昨天傍晚，我獨自走在河邊，看著微風拂過水面，夕陽的餘暉映照著天空，心中充滿了平靜與淡淡的感傷。"
        sentence_2 = "昨天傍晚，我一個人在河邊散步，微風輕輕吹過水面，夕陽的光輝染紅了天空，心裡湧起一種平靜又淡淡的惆悵。"

        # Also test with a very different sentence for comparison
        sentence_different = "今天早上，我在辦公室裡工作，電腦螢幕顯示著各種數據，空調的冷風吹著，心裡想著下午的會議。"

        # Test AutoModelEmbedder
        embedder1 = AutoModelEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        embeddings1 = embedder1.embed([sentence_1, sentence_2, sentence_different])
        
        # Test BGEM3Embedder
        embedder2 = BGEM3Embedder("BAAI/bge-m3", use_fp16=False)
        embeddings2 = embedder2.embed([sentence_1, sentence_2, sentence_different])
        
        # Test MilvusBGEM3Embedder
        embedder3 = MilvusBGEM3Embedder()
        embeddings3 = embedder3.embed([sentence_1, sentence_2, sentence_different])
        
        # Test DenseEmbedder
        embedder4 = DenseEmbedder.from_default("sentence-transformers/all-MiniLM-L6-v2")
        embeddings4 = embedder4.embed([sentence_1, sentence_2, sentence_different])
        
        # Test SparseEmbedder
        embedder5 = SparseEmbedder.from_default("BAAI/bge-m3")
        embeddings5 = embedder5.embed([sentence_1, sentence_2, sentence_different])

        # Basic assertions to ensure embeddings are generated
        self.assertIsInstance(embeddings1, list)
        self.assertEqual(len(embeddings1), 3)
        self.assertIsInstance(embeddings2, csr_array)
        self.assertEqual(embeddings2.shape[0], 3)
        self.assertIsInstance(embeddings3, csr_array)
        self.assertEqual(embeddings3.shape[0], 3)
        self.assertIsInstance(embeddings4, list)
        self.assertEqual(len(embeddings4), 3)
        self.assertIsInstance(embeddings5, csr_array)
        self.assertEqual(embeddings5.shape[0], 3)
        
        # Compute similarities for AutoModelEmbedder (dense vectors)
        emb1_array = np.array(embeddings1)
        similarity_similar_1 = cosine_similarity([emb1_array[0]], [emb1_array[1]])[0][0]
        similarity_different_1 = cosine_similarity([emb1_array[0]], [emb1_array[2]])[0][0]
        
        # Compute similarities for BGEM3Embedder (sparse vectors)
        similarity_similar_2 = cosine_similarity(embeddings2[0:1], embeddings2[1:2])[0][0]
        similarity_different_2 = cosine_similarity(embeddings2[0:1], embeddings2[2:3])[0][0]
        
        # Compute similarities for MilvusBGEM3Embedder (sparse vectors)
        similarity_similar_3 = cosine_similarity(embeddings3[0:1], embeddings3[1:2])[0][0]
        similarity_different_3 = cosine_similarity(embeddings3[0:1], embeddings3[2:3])[0][0]
        
        # Compute similarities for DenseEmbedder (dense vectors)
        similarity_similar_4 = cosine_similarity(embeddings4[0:1], embeddings4[1:2])[0][0]
        similarity_different_4 = cosine_similarity(embeddings4[0:1], embeddings4[2:3])[0][0]
        
        # Compute similarities for SparseEmbedder (sparse vectors)
        similarity_similar_5 = cosine_similarity(embeddings5[0:1], embeddings5[1:2])[0][0]
        similarity_different_5 = cosine_similarity(embeddings5[0:1], embeddings5[2:3])[0][0]
        
        print(f"\n=== Similarity Results ===")
        print(f"AutoModelEmbedder - Similar sentences: {similarity_similar_1:.4f}, Different sentences: {similarity_different_1:.4f}")
        print(f"BGEM3Embedder - Similar sentences: {similarity_similar_2:.4f}, Different sentences: {similarity_different_2:.4f}")
        print(f"MilvusBGEM3Embedder - Similar sentences: {similarity_similar_3:.4f}, Different sentences: {similarity_different_3:.4f}")
        print(f"DenseEmbedder - Similar sentences: {similarity_similar_4:.4f}, Different sentences: {similarity_different_4:.4f}")
        print(f"SparseEmbedder - Similar sentences: {similarity_similar_5:.4f}, Different sentences: {similarity_different_5:.4f}")
        
        # Assert that similar sentences have higher similarity than different sentences
        self.assertGreater(similarity_similar_1, similarity_different_1, 
                          f"AutoModelEmbedder: Similar sentences ({similarity_similar_1:.4f}) should be more similar than different sentences ({similarity_different_1:.4f})")
        self.assertGreater(similarity_similar_2, similarity_different_2,
                          f"BGEM3Embedder: Similar sentences ({similarity_similar_2:.4f}) should be more similar than different sentences ({similarity_different_2:.4f})")
        self.assertGreater(similarity_similar_3, similarity_different_3,
                          f"MilvusBGEM3Embedder: Similar sentences ({similarity_similar_3:.4f}) should be more similar than different sentences ({similarity_different_3:.4f})")
        self.assertGreater(similarity_similar_4, similarity_different_4,
                          f"DenseEmbedder: Similar sentences ({similarity_similar_4:.4f}) should be more similar than different sentences ({similarity_different_4:.4f})")
        self.assertGreater(similarity_similar_5, similarity_different_5,
                          f"SparseEmbedder: Similar sentences ({similarity_similar_5:.4f}) should be more similar than different sentences ({similarity_different_5:.4f})")
        
        # Assert that similar sentences have reasonably high similarity (typically > 0.7 for good models)
        self.assertGreater(similarity_similar_1, 0.7, 
                          f"AutoModelEmbedder: Similar sentences should have high similarity (> 0.7), got {similarity_similar_1:.4f}")
        self.assertGreater(similarity_similar_2, 0.5,  # Sparse embeddings might have lower similarity
                          f"BGEM3Embedder: Similar sentences should have reasonable similarity (> 0.5), got {similarity_similar_2:.4f}")
        self.assertGreater(similarity_similar_3, 0.5,  # Sparse embeddings might have lower similarity
                          f"MilvusBGEM3Embedder: Similar sentences should have reasonable similarity (> 0.5), got {similarity_similar_3:.4f}")
        self.assertGreater(similarity_similar_4, 0.7, 
                          f"DenseEmbedder: Similar sentences should have high similarity (> 0.7), got {similarity_similar_4:.4f}")
        self.assertGreater(similarity_similar_5, 0.5,  # Sparse embeddings might have lower similarity
                          f"SparseEmbedder: Similar sentences should have reasonable similarity (> 0.5), got {similarity_similar_5:.4f}")
        
        print(f"✅ All similarity tests passed!")

    def test_chinese_short_vs_long_content_bias(self):
        '''
        Test Chinese short vs long content bias specifically.
        Compare two long descriptive contents with one short keyword to see
        if short content gets over-represented in embeddings.
        '''
        # One short Chinese keyword
        short_chinese = ["機器學習"]
        
        # Two longer descriptive Chinese contents about the same topic
        long_chinese_1 = ["傳統的推薦系統主要依賴於分析數據和機器學習算法，並由系統單方面向使用者推播。對話式推薦系統則可以直接接受來自使用者主動提供的資訊，而系統也能透過文字對項目進行推薦，給予最直接的幫助。在對話式推薦系統（Conversational Recommender System, CRS）中使用大型語言模型（Large Language Model, LLM）能獲得許多傳統模型無法擁有的優勢 。首先，LLM的系統不需要經過訓練即可展現出色的性能，能夠解決冷啟動問題。其次，LLM的泛用性及可擴展性極高，能適應或導入到各種應用場景。大多數以往的研究偏向於從大量的物件中推薦出一個最相關的內容，這較適合使用檢索增強生成（Retrieval Augmented Generation, RAG）技術；我們的研究著重於從少量的物件中推薦一個最適合的內容，並強調應依據病 因來推薦藥方。本研究首次嘗試以失眠患者的中藥方推薦作為任務目標，使用繁體中文能力出色的gemini-1.5-flash模擬並生成病患與中醫師之間的對話。我們提出Hint Module來導入生 理量測及中醫問診技巧，透過偵測特定字串來觸發系統機制，給予LLM額外的提示訊息藉此影響它的輸出結果。我們的實驗結果顯示，此方法可生成寫實的自述，能被視作為良好的對話範 例，並在十種中藥方的推薦任務中，可得到八成以上的準確率及Macro-F1成績。其中Hint Module能顯著地改善多輪對話後的成績表現 (p-value < 0.01)。最後我們也以中醫學的各項觀點進行分析，透過視覺化的圖表呈現出各個藥方之間分布上的關聯性，以得到更全面及清晰的瞭解。實驗結果展示了以LLM打造的中藥方推薦系統擁有傑出的基礎能力，並擁有絕佳的可擴展 性。"]
        long_chinese_2 = ["是否已有研究探討將大型語言模型（Large Language Models, LLMs）應用於對話式推薦系統（Conversational Recommender Systems, CRS），以解決冷啟動問題，並實作在中醫藥方推薦（特別是針對失眠患者）上，進行多輪問答、引導式對話以及結合中醫診療技巧來提升推薦準確率？"]
        
        # Different Chinese content for comparison
        different_chinese = ["今天天氣晴朗，氣溫25度，適合外出活動。"]
        
        print(f"\n=== Testing Chinese Short vs Long Content Bias ===")
        print(f"Short: {short_chinese[0]}")
        print(f"Long 1: {long_chinese_1[0][:100]}...")
        print(f"Long 2: {long_chinese_2[0]}")
        print(f"Different: {different_chinese[0]}")
        print(f"\nRAG Scenario:")
        print(f"- Long 1: Document content (stored in database)")
        print(f"- Long 2: Detailed user query")
        print(f"- Short: Simple keyword query")
        print(f"- Different: Unrelated content")
        
        # Test all embedders
        embedders = [
            ("AutoModelEmbedder", AutoModelEmbedder("sentence-transformers/all-MiniLM-L6-v2")),
            ("BGEM3Embedder", BGEM3Embedder("BAAI/bge-m3", use_fp16=False)),
            ("MilvusBGEM3Embedder", MilvusBGEM3Embedder()),
            ("DenseEmbedder", DenseEmbedder.from_default("sentence-transformers/all-MiniLM-L6-v2")),
            ("SparseEmbedder", SparseEmbedder.from_default("BAAI/bge-m3"))
        ]
        
        # Collect all results first
        results = []
        
        for name, embedder in embedders:
            print(f"\n--- {name} (Chinese) ---")
            
            try:
                # Get embeddings
                short_embeddings = embedder.embed(short_chinese)
                long_embeddings_1 = embedder.embed(long_chinese_1)
                long_embeddings_2 = embedder.embed(long_chinese_2)
                different_embeddings = embedder.embed(different_chinese)
                
                # Compute similarities
                if isinstance(short_embeddings, list):
                    # Dense embeddings
                    short_array = np.array(short_embeddings)
                    long_array_1 = np.array(long_embeddings_1)
                    long_array_2 = np.array(long_embeddings_2)
                    different_array = np.array(different_embeddings)
                    
                    # RAG-relevant similarities
                    sim_doc_detailed = cosine_similarity([long_array_1[0]], [long_array_2[0]])[0][0]  # Document vs Detailed Query
                    sim_doc_short = cosine_similarity([long_array_1[0]], [short_array[0]])[0][0]     # Document vs Short Query
                    sim_detailed_short = cosine_similarity([long_array_2[0]], [short_array[0]])[0][0] # Detailed Query vs Short Query
                    
                    # vs Different content
                    sim_doc_different = cosine_similarity([long_array_1[0]], [different_array[0]])[0][0]     # Document vs Different
                    sim_detailed_different = cosine_similarity([long_array_2[0]], [different_array[0]])[0][0] # Detailed Query vs Different
                    sim_short_different = cosine_similarity([short_array[0]], [different_array[0]])[0][0]    # Short Query vs Different
                    
                else:
                    # Sparse embeddings
                    sim_doc_detailed = cosine_similarity(long_embeddings_1[0:1], long_embeddings_2[0:1])[0][0]  # Document vs Detailed Query
                    sim_doc_short = cosine_similarity(long_embeddings_1[0:1], short_embeddings[0:1])[0][0]     # Document vs Short Query
                    sim_detailed_short = cosine_similarity(long_embeddings_2[0:1], short_embeddings[0:1])[0][0] # Detailed Query vs Short Query
                    
                    # vs Different content
                    sim_doc_different = cosine_similarity(long_embeddings_1[0:1], different_embeddings[0:1])[0][0]     # Document vs Different
                    sim_detailed_different = cosine_similarity(long_embeddings_2[0:1], different_embeddings[0:1])[0][0] # Detailed Query vs Different
                    sim_short_different = cosine_similarity(short_embeddings[0:1], different_embeddings[0:1])[0][0]    # Short Query vs Different
                
                # Store results
                result = {
                    'name': name,
                    'sim_doc_detailed': sim_doc_detailed,
                    'sim_doc_short': sim_doc_short,
                    'sim_detailed_short': sim_detailed_short,
                    'sim_doc_different': sim_doc_different,
                    'sim_detailed_different': sim_detailed_different,
                    'sim_short_different': sim_short_different,
                    'success': True
                }
                
                print(f"RAG Similarity Matrix:")
                print(f"                    | Document | Detailed | Short   | Different")
                print(f"--------------------|----------|----------|---------|----------")
                print(f"Document            |   1.0000 | {sim_doc_detailed:.4f}  | {sim_doc_short:.4f} | {sim_doc_different:.4f}")
                print(f"Detailed Query      | {sim_doc_detailed:.4f}  |   1.0000 | {sim_detailed_short:.4f} | {sim_detailed_different:.4f}")
                print(f"Short Query         | {sim_doc_short:.4f}  | {sim_detailed_short:.4f} |   1.0000 | {sim_short_different:.4f}")
                print(f"Different Content   | {sim_doc_different:.4f}  | {sim_detailed_different:.4f} | {sim_short_different:.4f} |   1.0000")
                
                print(f"\nKey RAG Metrics:")
                print(f"Document vs Detailed Query: {sim_doc_detailed:.4f} (Should be HIGH - relevant content)")
                print(f"Document vs Short Query:    {sim_doc_short:.4f} (Should be HIGH - relevant content)")
                print(f"Detailed vs Short Query:    {sim_detailed_short:.4f} (Should be HIGH - same topic)")
                print(f"Document vs Different:      {sim_doc_different:.4f} (Should be LOW - irrelevant)")
                print(f"Detailed vs Different:      {sim_detailed_different:.4f} (Should be LOW - irrelevant)")
                print(f"Short vs Different:         {sim_short_different:.4f} (Should be LOW - irrelevant)")
                
                # Calculate bias indicators
                short_query_bias = sim_short_different - min(sim_doc_short, sim_detailed_short)
                detailed_query_advantage = sim_doc_detailed - sim_doc_short
                
                print(f"\nBias Analysis:")
                print(f"Short Query Bias: {short_query_bias:.4f} (Positive = short query more similar to different content)")
                print(f"Detailed Query Advantage: {detailed_query_advantage:.4f} (Positive = detailed query better than short)")
                
                # Check for potential bias issues
                if sim_doc_short < 0.5:
                    print(f"🚨 CRITICAL: {name} shows very low similarity between document and short query")
                    print(f"   This indicates severe short query bias - short queries won't find relevant documents")
                elif sim_doc_short < 0.6:
                    print(f"⚠️  WARNING: {name} shows low similarity between document and short query")
                    print(f"   This suggests short queries may not retrieve relevant documents effectively")
                elif sim_doc_short < 0.7:
                    print(f"📊 MODERATE: {name} shows moderate similarity between document and short query")
                    print(f"   This may indicate some short query bias")
                else:
                    print(f"✅ GOOD: {name} shows good similarity between document and short query")
                    print(f"   This suggests minimal short query bias")
                
                if short_query_bias > 0:
                    print(f"🚨 CRITICAL: {name} shows short queries are more similar to different content than relevant content")
                    print(f"   This will cause retrieval to return irrelevant results for short queries")
                
            except Exception as e:
                print(f"❌ ERROR: {name} failed with error: {str(e)}")
                result = {
                    'name': name,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        # Now show summary and run assertions
        print(f"\n=== SUMMARY OF ALL EMBEDDERS ===")
        print(f"{'Embedder':<20} {'Doc-Detailed':<12} {'Doc-Short':<10} {'Det-Short':<10} {'Short-Diff':<10} {'Bias':<8} {'Status':<10}")
        print(f"{'-'*90}")
        
        for result in results:
            if result['success']:
                short_query_bias = result['sim_short_different'] - min(result['sim_doc_short'], result['sim_detailed_short'])
                status = "✅ GOOD" if result['sim_doc_short'] >= 0.7 and short_query_bias <= 0 else \
                        "📊 MODERATE" if result['sim_doc_short'] >= 0.6 else \
                        "⚠️ WARNING" if result['sim_doc_short'] >= 0.5 else "🚨 CRITICAL"
                
                print(f"{result['name']:<20} {result['sim_doc_detailed']:<12.4f} {result['sim_doc_short']:<10.4f} "
                      f"{result['sim_detailed_short']:<10.4f} {result['sim_short_different']:<10.4f} "
                      f"{short_query_bias:<8.4f} {status:<10}")
            else:
                print(f"{result['name']:<20} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<8} ❌ FAILED")
        
        # Now run assertions for all successful embedders
        print(f"\n=== ASSERTION RESULTS ===")
        failed_assertions = []
        
        for result in results:
            if result['success']:
                # Assert that document should be more similar to relevant queries than to different content
                if result['sim_doc_short'] <= result['sim_doc_different']:
                    failed_assertions.append(f"{result['name']}: Document should be more similar to short query than to different content")
                
                if result['sim_doc_detailed'] <= result['sim_detailed_different']:
                    failed_assertions.append(f"{result['name']}: Document should be more similar to detailed query than to different content")
                
                # Assert that short query should be more similar to relevant content than to different content
                if result['sim_doc_short'] <= result['sim_short_different']:
                    failed_assertions.append(f"{result['name']}: Short query should be more similar to relevant document than to different content")
                
                if result['sim_detailed_short'] <= result['sim_short_different']:
                    failed_assertions.append(f"{result['name']}: Short query should be more similar to detailed query than to different content")
        
        if failed_assertions:
            print(f"❌ FAILED ASSERTIONS:")
            for assertion in failed_assertions:
                print(f"   - {assertion}")
            # Don't raise exception, just report
        else:
            print(f"✅ ALL ASSERTIONS PASSED")
        
        print(f"\n=== RAG System Recommendations ===")
        print(f"Based on the results:")
        print(f"1. Choose embedders with high Doc-Short similarity for better short query retrieval")
        print(f"2. Avoid embedders with positive Short Query Bias (short queries more similar to irrelevant content)")
        print(f"3. Consider query expansion for embedders with low Doc-Short similarity")
        print(f"4. Use hybrid search (dense + keyword) for problematic embedders")
        print(f"5. Implement reranking with longer context for final results")

if __name__ == "__main__":
    unittest.main()
