import unittest
from src.core.reranker import Reranker, IdentityReranker, AutoModelReranker
from unittest.mock import MagicMock, patch

# Minimal mock Document class for reranker tests
class MockField:
    def __init__(self, name, contents):
        self.name = name
        self.contents = contents

class MockDocument:
    def __init__(self, key, content_dict):
        self._key = key
        self._content = content_dict
    def key(self):
        return self._key
    def content(self):
        return self._content

class TestReranker(unittest.TestCase):
    def setUp(self):
        self.docs = [
            MockDocument("doc1", {"title": MockField("title", ["A"])}),
            MockDocument("doc2", {"title": MockField("title", ["B"])}),
            MockDocument("doc3", {"title": MockField("title", ["C"])}),
        ]
        self.query = "test query"

    def test_identity_reranker_returns_same_order(self):
        reranker = IdentityReranker()
        result = reranker.rerank(self.query, self.docs)
        self.assertEqual(result, self.docs)

    def test_identity_reranker_empty(self):
        reranker = IdentityReranker()
        result = reranker.rerank(self.query, [])
        self.assertEqual(result, [])

    @patch("src.core.reranker.AutoTokenizer.from_pretrained")
    @patch("src.core.reranker.AutoModelForSequenceClassification.from_pretrained")
    def test_automodel_reranker_sorts_by_score(self, mock_model_cls, mock_tokenizer_cls):
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1,2,3]], "attention_mask": [[1,1,1]]}
        mock_tokenizer_cls.return_value = mock_tokenizer
        # Mock model
        mock_model = MagicMock()
        # Simulate logits: doc2 highest, doc1 middle, doc3 lowest
        mock_logits = MagicMock()
        mock_scores = MagicMock()
        mock_scores.tolist.return_value = [0.5, 2.0, -1.0]
        mock_logits.view.return_value.float.return_value = mock_scores
        mock_model.return_value = MagicMock(logits=mock_logits)
        mock_model.eval.return_value = None
        mock_model.to.return_value = None
        mock_model_cls.return_value = mock_model

        reranker = AutoModelReranker(model_name="mock-model", device="cpu")
        # Patch _doc_to_string to avoid real doc parsing
        reranker._doc_to_string = lambda doc: doc.key()
        # Patch tokenizer to return a dummy tensor dict
        reranker.tokenizer = MagicMock(return_value=MagicMock(to=lambda device: {"input_ids": [[1,2,3]], "attention_mask": [[1,1,1]]}))
        # Patch model to return our mock logits
        reranker.model = MagicMock()
        reranker.model.eval.return_value = None
        reranker.model.to.return_value = None
        reranker.model.return_value = MagicMock(logits=mock_logits)
        # Patch torch.no_grad to a context manager that does nothing
        import contextlib
        reranker.__class__.__module__ = "src.core.reranker"  # for patching torch
        with patch("src.core.reranker.torch.no_grad", contextlib.nullcontext):
            result = reranker.rerank(self.query, self.docs)
        # Should be sorted by score: doc2 (2.0), doc1 (0.5), doc3 (-1.0)
        self.assertEqual([d.key() for d in result], ["doc2", "doc1", "doc3"])

    def test_reranker_interface_raises(self):
        reranker = Reranker()
        with self.assertRaises(NotImplementedError):
            reranker.rerank(self.query, self.docs)

if __name__ == "__main__":
    unittest.main() 