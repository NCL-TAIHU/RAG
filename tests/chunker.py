import unittest
from src.core.chunker import BaseChunker, LengthChunker, SentenceChunker, ChunkerMetaData


class TestChunker(unittest.TestCase):
    def setUp(self):
        """
        Common test documents for all chunkers.
        """
        self.long_text = "abcdefghijklmnopqrstuvwxyz" * 10  # 260 chars
        self.en_text = "This is sentence one. This is sentence two. This is sentence three."
        self.zh_text = "這是第一句。這是第二句。這是第三句。"
        self.docs = [self.long_text, self.en_text, self.zh_text]

    # ----------- 🔴 BaseChunker Abstract -----------
    def test_base_chunker_cannot_be_instantiated(self):
        """BaseChunker should not be instantiable directly."""
        with self.assertRaises(TypeError):
            BaseChunker()

    # ----------- 🔵 LengthChunker Tests -----------
    def test_length_chunker_chunk_splits_correctly(self):
        """LengthChunker splits text into fixed-size chunks."""
        chunker = LengthChunker(chunk_length=50)
        result = chunker.chunk([self.long_text])
        # Check number of chunks
        expected_chunks = len(self.long_text) // 50 + (1 if len(self.long_text) % 50 else 0)
        self.assertEqual(len(result[0]), expected_chunks)
        # Check chunk size
        for chunk in result[0][:-1]:
            self.assertEqual(len(chunk), 50)
        print("[Test] LengthChunker splitting passed.")

    def test_length_chunker_metadata(self):
        """LengthChunker metadata returns correct info."""
        chunker = LengthChunker(chunk_length=128)
        meta = chunker.metadata()
        self.assertIsInstance(meta, ChunkerMetaData)
        self.assertEqual(meta.chunker_type, "length")
        self.assertEqual(meta.params["chunk_length"], 128)
        print("[Test] LengthChunker metadata passed.")

    # ----------- 🟢 SentenceChunker Tests -----------
    def test_sentence_chunker_en_splits_correctly(self):
        """SentenceChunker splits English text by periods."""
        chunker = SentenceChunker(language="en")
        result = chunker.chunk([self.en_text])
        expected_sentences = ["This is sentence one", "This is sentence two", "This is sentence three", ""]
        self.assertEqual(result[0], expected_sentences)
        print("[Test] SentenceChunker EN splitting passed.")

    def test_sentence_chunker_zh_splits_correctly(self):
        """SentenceChunker splits Chinese text by full stop."""
        chunker = SentenceChunker(language="zh")
        result = chunker.chunk([self.zh_text])
        expected_sentences = ["這是第一句", "這是第二句", "這是第三句", ""]
        self.assertEqual(result[0], expected_sentences)
        print("[Test] SentenceChunker ZH splitting passed.")

    def test_sentence_chunker_invalid_language_raises(self):
        """SentenceChunker raises error for unsupported languages."""
        with self.assertRaises(AssertionError):
            SentenceChunker(language="fr")
        print("[Test] SentenceChunker invalid language check passed.")

    def test_sentence_chunker_metadata(self):
        """SentenceChunker metadata returns correct info."""
        chunker = SentenceChunker(language="en")
        meta = chunker.metadata()
        self.assertIsInstance(meta, ChunkerMetaData)
        self.assertEqual(meta.chunker_type, "sentence")
        self.assertEqual(meta.params["language"], "en")
        print("[Test] SentenceChunker metadata passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
