from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm



def csr_to_triples(csr_mat):
    coo = csr_mat.tocoo()
    return np.vstack([coo.row, coo.col, coo.data]).T


def sample_negatives(
    interactions: pl.DataFrame,
    num_negatives: int = 5,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pl.DataFrame:
    """Sample unobserved popular items for every user.

    The generated file is a local cache and must not be committed to Git.
    """
    rng = np.random.default_rng(seed)

    popular_items = (
        interactions.group_by("item_id")
        .len()
        .sort("len", descending=True)
        .head(100_000)["item_id"]
        .to_numpy()
    )
    user_items = (
        interactions.group_by("user_id")
        .agg(pl.col("item_id").alias("seen_items"))
        .iter_rows(named=True)
    )

    negatives = []
    for row in tqdm(user_items, desc="Sampling negatives"):
        candidates = popular_items[~np.isin(popular_items, row["seen_items"])]
        selected = rng.choice(
            candidates,
            size=min(num_negatives, len(candidates)),
            replace=False,
        )
        negatives.extend((row["user_id"], item, 0) for item in selected)

    sampled = pl.DataFrame(
        negatives,
        schema={"user_id": pl.Int64, "item_id": pl.Int64, "target": pl.Int8},
        orient="row",
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sampled.write_parquet(path)
    return sampled


if __name__ == "__main__":
    from implicit.datasets.reddit import get_reddit

    data = get_reddit()
    triples = csr_to_triples(data)
    interactions_df = pl.DataFrame(
        triples,
        schema={"user_id": pl.Int64, "item_id": pl.Int64, "target": pl.Int8},
    )
    sample_negatives(interactions_df, output_path="data/negative_pairs.parquet")
