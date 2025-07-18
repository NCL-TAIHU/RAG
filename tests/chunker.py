import unittest
from src.core.chunker import BaseChunker, LengthChunker, SentenceChunker, ChunkerMetaData


class TestChunker(unittest.TestCase):
    def setUp(self):
        """
        Common test documents for all chunkers.
        """
        self.long_text = "賽車運動是一項科技與刺激結合的文化，為何賽車運動歷久不衰，主要人類為了追求超越極限的成就感。然而，賽車場是賽車運動最重要的基底，臺灣在大鵬灣國際賽車場落成以後，賽車運動有了大幅變化，賽事整體提升了一個層級，引進了數量的高級跑車，提高整個臺灣民眾對於賽車文化上的參與度。進而提升臺灣車輛產業的經濟。透過本文探討賽車場與賽事經營管理模式，對相似國際賽車場進行案例研究，運用分析賽事營運品牌化差異來找出問題與建議。透過文獻與訪談，了解大鵬灣國際賽車場經營上之每年營業收入成本、賽事營業收入、賽車手、觀眾參與度統計及策劃上問題。讓臺灣賽車運動能夠永續經營，有朝能夠成為臺灣代表性文化之一。"  # 260 chars
        self.en_text = "This is sentence one. This is sentence two. This is sentence three."
        self.zh_text = "這是第一句。這是第二句。這是第三句。"
        self.docs = [self.long_text, self.en_text, self.zh_text]

    # # ----------- 🔴 BaseChunker Abstract -----------
    # def test_base_chunker_cannot_be_instantiated(self):
    #     """BaseChunker should not be instantiable directly."""
    #     with self.assertRaises(TypeError):
    #         BaseChunker()

    # # ----------- 🔵 LengthChunker Tests -----------
    # def test_length_chunker_chunk_splits_correctly(self):
    #     """LengthChunker splits text into fixed-size chunks."""
    #     chunker = LengthChunker(chunk_length=50)
    #     result = chunker.chunk([self.long_text])
    #     # Check number of chunks
    #     expected_chunks = len(self.long_text) // 50 + (1 if len(self.long_text) % 50 else 0)
    #     self.assertEqual(len(result[0]), expected_chunks)
    #     # Check chunk size
    #     for chunk in result[0][:-1]:
    #         self.assertEqual(len(chunk), 50)
    #     print("[Test] LengthChunker splitting passed.")

    # def test_length_chunker_metadata(self):
    #     """LengthChunker metadata returns correct info."""
    #     chunker = LengthChunker(chunk_length=128)
    #     meta = chunker.metadata()
    #     self.assertIsInstance(meta, ChunkerMetaData)
    #     self.assertEqual(meta.chunker_type, "length")
    #     self.assertEqual(meta.params["chunk_length"], 128)
    #     print("[Test] LengthChunker metadata passed.")

    # # ----------- 🟢 SentenceChunker Tests -----------
    # def test_sentence_chunker_en_splits_correctly(self):
    #     """SentenceChunker splits English text by periods."""
    #     chunker = SentenceChunker(language="en")
    #     result = chunker.chunk([self.en_text])
    #     expected_sentences = ["This is sentence one", "This is sentence two", "This is sentence three", ""]
    #     self.assertEqual(result[0], expected_sentences)
    #     print("[Test] SentenceChunker EN splitting passed.")

    # def test_sentence_chunker_zh_splits_correctly(self):
    #     """SentenceChunker splits Chinese text by full stop."""
    #     chunker = SentenceChunker(language="zh")
    #     result = chunker.chunk([self.zh_text])
    #     expected_sentences = ["這是第一句", "這是第二句", "這是第三句", ""]
    #     self.assertEqual(result[0], expected_sentences)
    #     print("[Test] SentenceChunker ZH splitting passed.")

    # def test_sentence_chunker_invalid_language_raises(self):
    #     """SentenceChunker raises error for unsupported languages."""
    #     with self.assertRaises(AssertionError):
    #         SentenceChunker(language="fr")
    #     print("[Test] SentenceChunker invalid language check passed.")

    # def test_sentence_chunker_metadata(self):
    #     """SentenceChunker metadata returns correct info."""
    #     chunker = SentenceChunker(language="en")
    #     meta = chunker.metadata()
    #     self.assertIsInstance(meta, ChunkerMetaData)
    #     self.assertEqual(meta.chunker_type, "sentence")
    #     self.assertEqual(meta.params["language"], "en")
    #     print("[Test] SentenceChunker metadata passed.")

    def test_chunk_length_chunker_bigger_than_text(self):
        """Chunker should return the text if the chunk size is bigger than the text."""
        chunker = LengthChunker(chunk_length=100)
        result = chunker.chunk([self.zh_text])
        self.assertEqual(result[0], [self.zh_text])
        print("[Test] LengthChunker chunk bigger than text passed.")

    def test_chunk_length_chunker_smaller_than_text(self):
        """Chunker should return the text if the chunk size is smaller than the text."""
        chunker = LengthChunker(chunk_length=10)
        result = chunker.chunk([self.zh_text])
        self.assertEqual(result[0], [self.zh_text])
        print("[Test] LengthChunker chunk smaller than text passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
