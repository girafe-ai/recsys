import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
from qdrant_client import QdrantClient, models

import operation


class OperationTests(unittest.TestCase):
    def test_reads_items_as_catalog_and_users_as_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "factors.h5"
            with h5py.File(path, "w") as data:
                data["items"] = [[1, 0], [0, 1], [1, 1]]
                data["users"] = [[2, 1], [1, 2]]
            result = operation.read_embeddings(path, 3)
        np.testing.assert_array_equal(result[0], [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(result[1], [[2, 1], [1, 2]])

    def test_failed_batch_stops_upload(self):
        client = Mock()
        client.upsert.side_effect = RuntimeError("upload failed")
        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            operation.upload_to_qdrant(client, "test", np.ones((3, 2)), 2)
        self.assertEqual(client.upsert.call_count, 1)

    def test_incomplete_collection_is_rejected(self):
        client = Mock()
        client.count.return_value.count = 2
        with self.assertRaisesRegex(RuntimeError, "2.*3"):
            operation.upload_to_qdrant(client, "test", np.ones((3, 2)), 2)

    def test_existing_collection_is_not_deleted_by_default(self):
        client = Mock()
        client.collection_exists.return_value = True
        with patch.object(operation, "QdrantClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "recreate"):
                operation.setup_qdrant("existing", 6333, 2)
        client.delete_collection.assert_not_called()

    def test_recall_counts_unique_ids_and_ignores_missing_ids(self):
        self.assertTrue(hasattr(operation, "recall_at_k"))
        truth = np.array([[2, 3, 5], [7, 8, 9]])
        found = np.array([[5, 2, 3], [7, 7, -1]])
        self.assertAlmostEqual(operation.recall_at_k(found, truth), 2 / 3)

    def test_qdrant_cosine_matches_normalized_flat_ip(self):
        items = np.array([[100, 0], [1, 1], [0, 3]], dtype="float32")
        query = np.array([1, 2], dtype="float32")
        client = QdrantClient(":memory:")
        try:
            client.create_collection(
                "test", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
            )
            operation.upload_to_qdrant(client, "test", items, 2)
            try:
                result = operation.search_qdrant(client, query, "test", 3)
            except AttributeError as error:
                self.fail(f"Search is incompatible with Qdrant client: {error}")
        finally:
            client.close()
        with tempfile.TemporaryDirectory() as folder:
            index = operation.build_faiss_index(items, str(Path(folder) / "flat.idx"))
            scores, ids = operation.search_faiss(index, query, 3)
        self.assertEqual([hit[1] for hit in result], ids.tolist())
        np.testing.assert_allclose([hit[0] for hit in result], scores, atol=1e-6)
        self.assertEqual(ids.tolist(), [1, 2, 0])

    @unittest.skipUnless(os.environ.get("QDRANT_TEST_URL"), "requires Qdrant server")
    def test_tiny_vectors_keep_cosine_scores_on_server(self):
        client = QdrantClient(url=os.environ["QDRANT_TEST_URL"])
        collection = "test_cosine_" + uuid.uuid4().hex
        try:
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
            )
            items = np.zeros((2, 128), dtype="float32")
            items[0, 0] = 2e-5
            items[1, :2] = [1, 1]
            original = items.copy()
            operation.upload_to_qdrant(client, collection, items, 2)
            result = operation.search_qdrant(client, items[0], collection, 2, exact=True)
            self.assertEqual([hit[1] for hit in result], [0, 1])
            np.testing.assert_allclose([hit[0] for hit in result], [1, 1 / np.sqrt(2)], atol=1e-6)
            np.testing.assert_array_equal(items, original)
        finally:
            client.delete_collection(collection)
            client.close()


if __name__ == "__main__":
    unittest.main()
