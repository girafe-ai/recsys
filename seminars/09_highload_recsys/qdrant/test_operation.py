import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
from qdrant_client import QdrantClient, models
from omegaconf.errors import ConfigKeyError

import operation


class OperationTests(unittest.TestCase):
    def test_quantization_settings_can_be_overridden_from_cli(self):
        try:
            config = operation.load_config([
                "quantization.type=scalar", "quantization.quantile=0.95",
                "quantization.rescore=false", "quantization.oversampling=4",
            ])
        except ConfigKeyError as error:
            self.fail(f"Quantization settings are missing: {error}")
        self.assertEqual(config.quantization.type, "scalar")
        self.assertEqual(config.quantization.quantile, 0.95)
        self.assertIs(config.quantization.rescore, False)
        self.assertEqual(config.quantization.oversampling, 4)

    def test_quantization_config_supports_each_mode(self):
        self.assertTrue(hasattr(operation, "build_quantization_config"))
        for kind in ["none", "scalar", "product", "binary"]:
            with self.subTest(kind=kind):
                config = operation.load_config([f"quantization.type={kind}"])
                result = operation.build_quantization_config(config.quantization)
                if kind == "none":
                    self.assertIsNone(result)
                else:
                    payload = result.model_dump(exclude_none=True)[kind]
                    self.assertIs(payload["always_ram"], True)
                    if kind == "scalar":
                        self.assertEqual(payload["type"], "int8")
                        self.assertEqual(payload["quantile"], 0.99)
                    elif kind == "product":
                        self.assertEqual(payload["compression"], "x8")

    def test_invalid_quantization_settings_are_rejected(self):
        self.assertTrue(hasattr(operation, "build_quantization_config"))
        for overrides in [
            ["quantization.type=unknown"],
            ["quantization.type=scalar", "quantization.quantile=0.1"],
            ["quantization.type=product", "quantization.compression=x3"],
        ]:
            with self.subTest(overrides=overrides):
                config = operation.load_config(overrides)
                with self.assertRaises(ValueError):
                    operation.build_quantization_config(config.quantization)

    def test_exact_search_disables_quantization(self):
        client = Mock()
        client.query_points.return_value.points = []
        operation.search_qdrant(client, np.ones(128), "test", exact=True)
        params = client.query_points.call_args.kwargs["search_params"]
        self.assertIsNotNone(params.quantization)
        self.assertIs(params.quantization.ignore, True)

    def test_search_passes_rescore_and_oversampling(self):
        client = Mock()
        client.query_points.return_value.points = []
        options = models.QuantizationSearchParams(ignore=False, rescore=False, oversampling=4)
        try:
            operation.search_qdrant(client, np.ones(128), "test", quantization=options)
        except TypeError as error:
            self.fail(f"Quantization search parameters are missing: {error}")
        params = client.query_points.call_args.kwargs["search_params"]
        self.assertIs(params.quantization.rescore, False)
        self.assertEqual(params.quantization.oversampling, 4)

    def test_collection_receives_quantization_config(self):
        client = Mock()
        client.collection_exists.return_value = False
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
        )
        with patch.object(operation, "QdrantClient", return_value=client):
            try:
                operation.setup_qdrant("test", 6333, 128, quantization=quantization)
            except TypeError as error:
                self.fail(f"Collection quantization is missing: {error}")
        self.assertEqual(client.create_collection.call_args.kwargs["quantization_config"], quantization)

    def test_cli_help_runs(self):
        result = subprocess.run(
            [sys.executable, operation.__file__, "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recreate", result.stdout)

    def test_config_overrides_preserve_types_and_reject_unknown_keys(self):
        self.assertTrue(hasattr(operation, "load_config"))
        config = operation.load_config(["num_queries=12", "recreate=true"])
        self.assertEqual(config.num_queries, 12)
        self.assertIs(config.recreate, True)
        with self.assertRaises(ConfigKeyError):
            operation.load_config(["unknown_option=1"])

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
    def test_quantized_search_and_exact_reference_on_server(self):
        rng = np.random.default_rng(17)
        items = rng.normal(size=(4096, 128)).astype("float32")
        queries = rng.normal(size=(4, 128)).astype("float32")
        index = operation.build_faiss_index(items)
        normalized_items = items / np.linalg.norm(items, axis=1, keepdims=True)
        client = QdrantClient(url=os.environ["QDRANT_TEST_URL"], timeout=60)
        try:
            for kind in ["scalar", "product", "binary"]:
                with self.subTest(kind=kind):
                    name = "test_quantization_" + uuid.uuid4().hex
                    config = operation.load_config([f"quantization.type={kind}"])
                    client.create_collection(
                        name,
                        vectors_config=models.VectorParams(size=128, distance=models.Distance.COSINE),
                        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10),
                        optimizers_config=models.OptimizersConfigDiff(
                            indexing_threshold=0, default_segment_number=2
                        ),
                        quantization_config=operation.build_quantization_config(config.quantization),
                    )
                    try:
                        operation.upload_to_qdrant(client, name, items, 1024)
                        operation.wait_for_index(client, name, 120)
                        info = client.get_collection(name)
                        self.assertIn(kind, info.config.quantization_config.model_dump())
                        for query in queries:
                            _, expected = operation.search_faiss(index, query, 10)
                            exact = operation.search_qdrant(
                                client, query, name, 10, exact=True,
                                quantization=models.QuantizationSearchParams(ignore=False, rescore=False),
                            )
                            self.assertEqual([hit[1] for hit in exact], expected.tolist())
                            for rescore in [False, True]:
                                hits = operation.search_qdrant(
                                    client, query, name, 10, hnsw_ef=256,
                                    quantization=models.QuantizationSearchParams(
                                        ignore=False, rescore=rescore, oversampling=2
                                    ),
                                )
                                ids = [hit[1] for hit in hits]
                                self.assertEqual(len(set(ids)), 10)
                                if rescore:
                                    expected_scores = normalized_items[ids] @ (query / np.linalg.norm(query))
                                    np.testing.assert_allclose(
                                        [hit[0] for hit in hits], expected_scores, atol=1e-5
                                    )
                    finally:
                        client.delete_collection(name)
        finally:
            client.close()

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
