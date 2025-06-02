import unittest
from src.core.data import DataLoader
from src.core.entity import Document

class TestDataLoaders(unittest.TestCase):
    def test_arxiv_dataloader(self):
        self._test_dataloader_runs("arxiv")

    def test_history_dataloader(self):
        self._test_dataloader_runs("history")

    def _test_dataloader_runs(self, dataset_name: str):
        dataloader = DataLoader.from_default(dataset_name)
        count = 0

        for batch in dataloader.load():
            self.assertIsInstance(batch, list)
            for doc in batch:
                self.assertIsInstance(doc, Document)
            count += 1
            if count >= 100:
                break

        self.assertGreater(count, 0, f"{dataset_name} dataloader did not yield any batches.")

if __name__ == "__main__":
    unittest.main()
