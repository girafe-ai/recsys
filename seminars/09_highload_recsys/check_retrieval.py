"""Check numerical correctness of the standalone notebook's retrieval helpers."""

import json
from pathlib import Path
import unittest

import numpy as np


NOTEBOOK = Path(__file__).with_name("seminar.ipynb")
NAMESPACE = {}
for cell in json.loads(NOTEBOOK.read_text())["cells"]:
    if "helpers" in cell.get("metadata", {}).get("tags", []):
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), NAMESPACE)


class RetrievalChecks(unittest.TestCase):
    def helper(self, name):
        self.assertTrue(name in NAMESPACE, f"Notebook is missing helper {name}")
        return NAMESPACE[name]

    def test_recall_ignores_rank_and_repeated_or_missing_ids(self):
        recall = self.helper("ann_recall")
        truth = np.array([[2, 3, 5], [7, 8, 9]])
        found = np.array([[5, 2, 3], [7, 7, -1]])
        self.assertAlmostEqual(recall(found, truth), 2 / 3)

    def test_normalization_restores_cosine_ranking_without_mutating_inputs(self):
        normalize = self.helper("normalize_embeddings")
        items = np.array([[100, 0], [1, 1], [0, 1]], dtype="float32")
        queries = np.array([[1, 2]], dtype="float32")
        original_items, original_queries = items.copy(), queries.copy()
        normalized_items, normalized_queries = normalize(items, queries)
        np.testing.assert_allclose(np.linalg.norm(normalized_items, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(normalized_queries, axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(items, original_items)
        np.testing.assert_array_equal(queries, original_queries)
        self.assertEqual(int(np.argmax(queries @ items.T)), 0)
        self.assertEqual(int(np.argmax(normalized_queries @ normalized_items.T)), 1)

    def test_normalization_keeps_zero_vectors_finite(self):
        normalize = self.helper("normalize_embeddings")
        items, queries = normalize(np.array([[0, 0], [3, 4]], dtype="float32"),
                                   np.array([[0, 0]], dtype="float32"))
        self.assertTrue(np.isfinite(items).all() and np.isfinite(queries).all())
        np.testing.assert_array_equal(items[0], [0, 0])
        np.testing.assert_array_equal(queries[0], [0, 0])

    def test_setting_selection_enforces_quality_before_comparing_latency(self):
        choose = self.helper("choose_setting")
        rows = [
            {'method': 'fast but inaccurate', 'ann_recall': .2, 'p95_ms': .01},
            {'method': 'qualified fast', 'ann_recall': .992, 'p95_ms': 2},
            {'method': 'qualified slow', 'ann_recall': .999, 'p95_ms': 3},
        ]
        self.assertEqual(choose(rows, .99)['method'], 'qualified fast')
        self.assertIsNone(choose(rows, 1.0))

    def test_topk_selects_objects_within_each_query(self):
        topk = self.helper("topk")
        scores = np.array([[1, 9, 3, 7], [12, 4, 10, 2]], dtype="float32")
        distances, ids = topk(scores, 2)
        np.testing.assert_array_equal(ids, [[1, 3], [0, 2]])
        np.testing.assert_array_equal(distances, [[9, 7], [12, 10]])

    def test_blocked_search_matches_full_sort_across_partial_blocks(self):
        exact = self.helper("exact_topk")
        rng = np.random.default_rng(8)
        items = rng.normal(size=(23, 7)).astype("float32")
        queries = rng.normal(size=(5, 7)).astype("float32")
        scores = queries @ items.T
        expected = np.argsort(-scores, axis=1)[:, :6]
        distances, ids = exact(queries, items, 6, user_block=2, item_block=4)
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(distances, np.take_along_axis(scores, expected, axis=1), atol=1e-5)

    def test_merge_discards_padding_even_with_negative_scores(self):
        merge = self.helper("merge_topk")
        parts = [
            (np.array([[-3, -8, 999]], dtype="float32"), np.array([[4, 7, -1]])),
            (np.array([[-1, -4]], dtype="float32"), np.array([[2, 9]])),
        ]
        distances, ids = merge(parts, 3)
        np.testing.assert_array_equal(ids, [[2, 4, 9]])
        np.testing.assert_array_equal(distances, [[-1, -3, -4]])

    def test_scalar_codes_are_bytes_and_constant_dimensions_survive(self):
        quantize = self.helper("scalar_quantize")
        restore = self.helper("scalar_restore")
        values = np.array([[-1, 7], [0, 7], [1, 7]], dtype="float32")
        low, high = values.min(axis=0), values.max(axis=0)
        codes = quantize(values, low, high)
        self.assertEqual(codes.dtype, np.uint8)
        restored = restore(codes, low, high)
        np.testing.assert_allclose(restored, values, atol=1 / 255 + 1e-6)

    def test_scalar_constant_range_clips_new_values_to_the_constant(self):
        quantize = self.helper("scalar_quantize")
        restore = self.helper("scalar_restore")
        low = high = np.array([7], dtype="float32")
        values = np.array([[-100], [7], [100]], dtype="float32")
        codes = quantize(values, low, high)
        np.testing.assert_array_equal(codes, np.zeros_like(codes))
        np.testing.assert_array_equal(restore(codes, low, high), np.full_like(values, 7))

    def test_quantile_edges_use_the_same_clipping_bounds_as_minmax(self):
        fit = self.helper("fit_quantile_edges")
        values = np.column_stack([np.arange(101), np.full(101, 7)]).astype("float32")
        edges = fit(values, levels=4)
        np.testing.assert_allclose(edges[:, 0], [1, 25.5, 50, 74.5, 99])
        np.testing.assert_array_equal(edges[:, 1], np.full(5, 7))

    def test_quantile_codes_select_intervals_and_restore_their_centers(self):
        quantize = self.helper("quantile_quantize")
        restore = self.helper("quantile_restore")
        edges = np.array([[-1, 7], [-.2, 7], [0, 7], [.2, 7], [1, 7]], dtype="float32")
        values = np.array([[-9, 7], [-.1, 7], [.15, 7], [9, 7]], dtype="float32")
        codes = quantize(values, edges)
        self.assertEqual(codes.dtype, np.uint8)
        np.testing.assert_array_equal(codes, [[0, 0], [1, 0], [2, 0], [3, 0]])
        restored = restore(codes, edges)
        np.testing.assert_allclose(restored, [[-.6, 7], [-.1, 7], [.1, 7], [.6, 7]], atol=1e-6)

    def test_quantile_clipping_skips_empty_boundary_intervals(self):
        quantize = self.helper("quantile_quantize")
        edges = np.array([[0], [0], [1], [2], [3]], dtype="float32")
        values = np.array([[-100], [0], [.5], [100]], dtype="float32")
        np.testing.assert_array_equal(quantize(values, edges), [[1], [1], [1], [3]])

    def test_reranking_uses_global_ids_and_recovers_exact_order(self):
        rerank = self.helper("rerank_candidates")
        items = np.array([[1, 0], [0, 2], [3, 0], [0, -1]], dtype="float32")
        queries = np.array([[1, 0.1], [0.1, 1]], dtype="float32")
        candidates = np.array([[1, 0, 2], [3, 0, 1]])
        distances, ids = rerank(queries, items, candidates, 2)
        np.testing.assert_array_equal(ids, [[2, 0], [1, 0]])
        np.testing.assert_allclose(distances, [[3, 1], [2, 0.1]])

    def test_parallel_and_sequential_shards_preserve_original_ids(self):
        build = self.helper("build_shards")
        search = self.helper("search_shards")
        rng = np.random.default_rng(19)
        items = rng.normal(size=(29, 8)).astype("float32")
        queries = rng.normal(size=(4, 8)).astype("float32")
        scores = queries @ items.T
        expected = np.argsort(-scores, axis=1)[:, :12]
        shards = build(items, 4, "flat")
        for parallel in [False, True]:
            distances, ids = search(shards, queries, 12, parallel=parallel)
            np.testing.assert_array_equal(ids, expected)
            np.testing.assert_allclose(distances, np.take_along_axis(scores, expected, axis=1), atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
