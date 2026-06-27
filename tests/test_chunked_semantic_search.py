import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cli.lib import semantic_search as semantic_search_module


class FakeModel:
    def encode(self, texts, show_progress_bar=False):
        return np.array([[float(len(text)), 0.0] for text in texts], dtype=float)


class ChunkedSemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.patcher = patch.object(semantic_search_module, "SentenceTransformer", FakeModel)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(os.chdir, self.original_cwd)
        self.addCleanup(self.temp_dir.cleanup)

    def test_build_chunk_embeddings_creates_metadata_and_saves_cache(self) -> None:
        search = semantic_search_module.ChunkedSemanticSearch()
        documents = [
            {
                "id": 1,
                "title": "Example Movie",
                "description": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
            }
        ]

        embeddings = search.build_chunk_embeddings(documents)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(search.chunk_metadata), 2)
        self.assertEqual(search.chunk_metadata[0]["movie_idx"], 0)
        self.assertEqual(search.chunk_metadata[1]["chunk_idx"], 1)
        self.assertTrue(os.path.exists("cache/chunk_embeddings.npy"))

        with open("cache/chunk_metadata.json", "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.assertEqual(metadata["total_chunks"], 2)

    def test_load_or_create_chunk_embeddings_uses_cache(self) -> None:
        search = semantic_search_module.ChunkedSemanticSearch()
        documents = [
            {
                "id": 1,
                "title": "Example Movie",
                "description": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
            }
        ]

        search.build_chunk_embeddings(documents)
        cached_search = semantic_search_module.ChunkedSemanticSearch()
        embeddings = cached_search.load_or_create_chunk_embeddings(documents)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(cached_search.chunk_metadata[0]["movie_idx"], 0)


if __name__ == "__main__":
    unittest.main()
