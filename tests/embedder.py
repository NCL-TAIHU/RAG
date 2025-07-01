import unittest
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr

from src.core.embedder import AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder  # adjust the import path

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

if __name__ == "__main__":
    unittest.main()
