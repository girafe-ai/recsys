from collections import defaultdict
import random

from implicit.datasets.reddit import get_reddit
import numpy as np
import polars as pl
from tqdm import tqdm



def csr_to_triples(csr_mat):
    coo = csr_mat.tocoo()
    return np.vstack([coo.row, coo.col, coo.data]).T


def sample_negatives(
    interactions: pl.DataFrame,
    num_negatives: int = 5,
    seed: int = 42
) -> pl.DataFrame:
    np.random.seed(seed)
    
    # Get top N most popular items
    popular_items = (
        interactions['item_id']
        .value_counts()
        .head(100_000)['item_id']  # Only consider top 100k items
        .to_numpy()
    )
    
    users = interactions['user_id'].unique().to_numpy()

    user_pos_items = (
        interactions
        .group_by('user_id')
        .agg(pl.col('item_id').alias('positives'))
    )
    
    # Generate negatives by sampling from popular items
    negatives = []
    print("Generate negatives...")
    for user in tqdm(users):
        user_pos = user_pos_items.filter(pl.col('user_id') == user)['positives'][0]
        
        candidates = np.setdiff1d(popular_items, user_pos, assume_unique=True)
        if len(candidates) > 0:
            selected = np.random.choice(
                candidates,
                size=min(num_negatives, len(candidates)),
                replace=False
            )
            negatives.extend([
                [user, item, 0]
                for item in selected
            ])
    
    df = pl.DataFrame(
        negatives,
        schema=['user_id', 'item_id', 'target']
    )

    df.write_parquet("negative_pairs.parquet")


if __name__ == "__main__":
    data = get_reddit()
    triples = csr_to_triples(data)
    positives_df = pl.DataFrame(triples, schema={"user_id": pl.Int64, "item_id": pl.Int64, "rating": pl.Int64})
    sample_negatives(positives_df)