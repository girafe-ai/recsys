"""Regression checks for the encoders defined in the self-contained notebook."""
import ast
import json
from pathlib import Path
import typing as tp
import unittest

import numpy as np
import polars as pl
import torch


def notebook_encoders():
    notebook = json.loads(Path(__file__).with_name('seminar.ipynb').read_text())
    scope = {'torch': torch, 'pl': pl, 'np': np, 'tp': tp}
    wanted = {'MultihashTransform', 'PiecewiseLinearEncodingTransform'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if not any(f'class {name}' in source for name in wanted):
            continue
        tree = ast.parse(source)
        definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook encoders>', 'exec'), scope)
    return scope['MultihashTransform'], scope['PiecewiseLinearEncodingTransform']


Hash, PLE = notebook_encoders()


class EncodingChecks(unittest.TestCase):
    def test_hash_breaks_the_old_modulo_equivalence(self):
        transform = Hash(1024, seeds=[[17, 31], [23, 47]])
        first = transform({'sparse': torch.tensor([123, 456])})['sparse']
        second = transform({'sparse': torch.tensor([1147, 1480])})['sparse']
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(((first >= 0) & (first < 1024)).all())

    def test_hash_batch_matches_individual_rows(self):
        transform = Hash(1024, seeds=[[17, 31], [23, 47]])
        values = torch.tensor([[123, 456], [567, 890]])
        batch = transform({'sparse': values})['sparse']
        rows = torch.stack([transform({'sparse': row})['sparse'] for row in values])
        np.testing.assert_allclose(batch, rows)

    def test_ple_constant_is_finite_zero(self):
        transform = PLE(pl.DataFrame({'x': [2., 2., 2.]}), n_bins=4)
        result = transform({'dense': torch.tensor([3.])})['dense']
        self.assertEqual(result.numel(), 1)
        np.testing.assert_allclose(result, torch.zeros(1))

    def test_ple_single_interval_keeps_one_coordinate(self):
        transform = PLE(pl.DataFrame({'x': [0., 1., 2.]}), n_bins=1)
        for x, expected in [(-1., -0.5), (0., 0.), (1., 0.5), (2., 1.), (3., 1.5)]:
            result = transform({'dense': torch.tensor([x])})['dense']
            np.testing.assert_allclose(result, torch.tensor([expected]))

    def test_ple_boundaries_extrapolation_and_padding(self):
        train = pl.DataFrame({'x': [0., 1., 2., 3., 4.], 'constant': [7.] * 5})
        transform = PLE(train, n_bins=2)
        for x, expected in [(-1., [-0.5, 0.]), (1., [0.5, 0.]), (2., [1., 0.]), (5., [1., 1.5])]:
            result = transform({'dense': torch.tensor([x, 7.])})['dense']
            np.testing.assert_allclose(result, torch.tensor(expected + [0., 0.]))
        batch = torch.tensor([[1., 7.], [5., 7.]])
        encoded = transform({'dense': batch})['dense']
        rows = torch.stack([transform({'dense': row})['dense'] for row in batch])
        np.testing.assert_allclose(encoded, rows)


if __name__ == '__main__':
    unittest.main()
