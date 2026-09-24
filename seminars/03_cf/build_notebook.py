"""Build the self-contained collaborative-filtering seminar notebook."""

import json
from pathlib import Path
from textwrap import dedent


cells = []


def add(kind, content):
    source = dedent(content).strip("\n") + "\n"
    cell = {
        "cell_type": kind,
        "id": f"cf-{len(cells):02d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cells.append(cell)


add(
    "markdown",
    r"""
    # Seminar 03: Collaborative filtering

    This seminar adapts the item-to-item and user-to-user examples from the `25s_msai` course branch. We will build recommendations from user ratings alone, compare two neighborhood models with a simple baseline, and discuss when the comparison is reliable.

    **Learning goals:** distinguish item-based from user-based CF; understand centering, overlap and shrinkage; avoid evaluation leakage; interpret RMSE/MAE and inspect recommendations.
    """,
)

add(
    "code",
    r"""
    # Run this cell first in Colab. The local `recsys` Conda environment already
    # contains these packages, so it does not install anything on your laptop.
    import sys
    import subprocess

    try:
        import google.colab  # noqa: F401
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib"],
            check=True,
        )

    from io import BytesIO
    from pathlib import Path
    from urllib.request import urlopen
    from zipfile import ZipFile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix

    RANDOM_STATE = 42
    if IN_COLAB:
        DATA_DIR = Path("/content/recsys_cf_data")
    else:
        NOTEBOOK_DIR = Path.cwd()
        if not (NOTEBOOK_DIR / "seminar.ipynb").exists():
            NOTEBOOK_DIR = Path.cwd() / "seminars" / "03_cf"
        DATA_DIR = NOTEBOOK_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data cache: {DATA_DIR}")
    """,
)

add(
    "markdown",
    r"""
    ## 1. Data and task

    We use [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): explicit movie ratings on a 1–5 scale. The first run downloads `u.data` and `u.item` from the official GroupLens archive. Later runs reuse the local cache.

    A missing rating means **unknown**, not zero or a dislike. We will predict ratings on interactions hidden from each user's history; recommendations are then generated only from items not present in that training history.
    """,
)

add(
    "code",
    r"""
    ratings_path = DATA_DIR / "u.data"
    movies_path = DATA_DIR / "u.item"
    # Reuse MovieLens 100K if seminar 01 / Surprise already cached it locally.
    previous_cache = Path.home() / ".surprise_data" / "ml-100k"
    if not ratings_path.exists() and (previous_cache / "u.data").exists():
        ratings_path = previous_cache / "u.data"
    if not movies_path.exists() and (previous_cache / "u.item").exists():
        movies_path = previous_cache / "u.item"
    if not (ratings_path.exists() and movies_path.exists()):
        archive_url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        with urlopen(archive_url, timeout=60) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            ratings_path.write_bytes(archive.read("ml-100k/u.data"))
            movies_path.write_bytes(archive.read("ml-100k/u.item"))

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
    )
    assert len(ratings) == 100_000
    assert ratings["rating"].between(1, 5).all()
    assert not ratings.duplicated(["user_id", "item_id"]).any()
    display(ratings.head())
    print(f"{ratings.user_id.nunique()} users, {ratings.item_id.nunique()} rated items")
    """,
)

add(
    "code",
    r"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout="constrained")
    ratings["rating"].value_counts().sort_index().plot.bar(ax=axes[0], color="#278f88")
    axes[0].set(title="Rating distribution", xlabel="Stars", ylabel="Interactions")
    ratings.groupby("user_id").size().plot.hist(ax=axes[1], bins=35, color="#278f88")
    axes[1].set(title="Ratings per user", xlabel="Count", ylabel="Users")
    ratings.groupby("item_id").size().plot.hist(ax=axes[2], bins=35, color="#278f88")
    axes[2].set(title="Ratings per item", xlabel="Count", ylabel="Items")
    plt.show()
    """,
)

add(
    "markdown",
    r"""
    ## 2. What the neighborhoods measure

    **Item-to-item:** two movies are similar when users tend to rate them above or below their *own* average in the same way. We use adjusted cosine similarity on user-centered ratings. **User-to-user:** two users are similar when their centered rating patterns align. This is cosine on centered vectors, which is **not exactly Pearson correlation computed only on co-rated items**.

    $s(a,b)=\frac{\sum_{u\in I_a\cap I_b}x_{ua}x_{ub}}{\sqrt{\sum_{u\in I_a}x_{ua}^2}\sqrt{\sum_{u\in I_b}x_{ub}^2}}\cdot\frac{n_{ab}}{n_{ab}+\lambda}$

    Here $x_{ui}=r_{ui}-\bar r_u$ on observed ratings, $n_{ab}$ is the number of shared observations, and $\lambda$ shrinks similarities supported by few users. We set similarities with fewer than three shared observations to zero. Negative similarity is valid mathematically, but this simple recommender uses only **positive** neighbors; it never treats a missing pair as a negative similarity.
    """,
)

add(
    "markdown",
    r"""
    ## 3. Hold out ratings without leakage

    We hide 20% of each user's interactions, leaving at least one training rating per user. User means, similarities and popularity are then learned **only from the training part**. The split is random, so this demonstrates offline rating prediction rather than a chronological production evaluation. Some held-out items can be absent from training; those cold items are excluded from the comparison and counted below.
    """,
)

add(
    "code",
    r"""
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(movies["item_id"].unique())
    ratings = ratings.copy()
    ratings["user_idx"] = np.searchsorted(user_ids, ratings["user_id"].to_numpy())
    ratings["item_idx"] = np.searchsorted(item_ids, ratings["item_id"].to_numpy())

    rng = np.random.default_rng(RANDOM_STATE)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for indices in ratings.groupby("user_idx").indices.values():
        holdout_count = min(max(1, round(0.2 * len(indices))), len(indices) - 1)
        test_mask[rng.choice(indices, size=holdout_count, replace=False)] = True

    train = ratings.loc[~test_mask].copy()
    test_all = ratings.loc[test_mask].copy()
    known_items = set(train["item_idx"])
    test = test_all.loc[test_all["item_idx"].isin(known_items)].copy()
    assert set(train.index).isdisjoint(test_all.index)
    assert train["user_idx"].nunique() == len(user_ids)
    print(f"Train: {len(train):,}; held out: {len(test_all):,}; evaluated: {len(test):,}")
    print(f"Cold-item held-out rows excluded: {len(test_all) - len(test)}")
    """,
)

add(
    "code",
    r"""
    n_users, n_items = len(user_ids), len(item_ids)
    user_count = np.bincount(train["user_idx"], minlength=n_users)
    user_sum = np.bincount(
        train["user_idx"], weights=train["rating"], minlength=n_users
    )
    global_mean = float(train["rating"].mean())
    user_mean = np.divide(
        user_sum,
        user_count,
        out=np.full(n_users, global_mean),
        where=user_count > 0,
    )
    rows = train["user_idx"].to_numpy()
    cols = train["item_idx"].to_numpy()
    values = train["rating"].to_numpy(dtype=float)
    residuals = values - user_mean[rows]
    shape = (n_users, n_items)
    train_ratings = csr_matrix((values, (rows, cols)), shape=shape)
    centered = csr_matrix((residuals, (rows, cols)), shape=shape)
    centered_csc = centered.tocsc()
    observed = csr_matrix((np.ones(len(train)), (rows, cols)), shape=shape)
    assert train_ratings.nnz == len(train)
    """,
)

add(
    "code",
    r"""
    def cosine_with_overlap(matrix, observation_matrix, axis, min_common=3, shrinkage=10):
        # Cosine similarity of rows or columns, weighted by shared observations.
        if axis == "item":
            gram = (matrix.T @ matrix).toarray()
            overlap = (observation_matrix.T @ observation_matrix).toarray()
        elif axis == "user":
            gram = (matrix @ matrix.T).toarray()
            overlap = (observation_matrix @ observation_matrix.T).toarray()
        else:
            raise ValueError("axis must be 'item' or 'user'")
        norms = np.sqrt(np.maximum(np.diag(gram), 0))
        denominator = np.outer(norms, norms)
        cosine = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
        similarity = cosine * overlap / (overlap + shrinkage)
        similarity[overlap < min_common] = 0
        np.fill_diagonal(similarity, 0)
        return similarity, overlap


    item_similarity, item_overlap = cosine_with_overlap(centered, observed, axis="item")
    user_similarity, user_overlap = cosine_with_overlap(centered, observed, axis="user")

    # A tiny hand-check: two centered item columns point in opposite directions.
    toy = csr_matrix([[0.5, -0.5], [0.5, -0.5]])
    toy_observed = csr_matrix(np.ones((2, 2)))
    toy_sim, _ = cosine_with_overlap(toy, toy_observed, "item", min_common=2, shrinkage=0)
    assert np.isclose(toy_sim[0, 1], -1.0)
    for similarity in (item_similarity, user_similarity):
        assert np.allclose(similarity, similarity.T)
        assert np.all(np.diag(similarity) == 0)
        assert np.isfinite(similarity).all()
        assert np.max(np.abs(similarity)) <= 1 + 1e-12
    print(f"Item matrix: {item_similarity.shape}; user matrix: {user_similarity.shape}")
    """,
)

add(
    "code",
    r"""
    def nearest_items(item_id, n=5):
        item_idx = int(np.searchsorted(item_ids, item_id))
        if item_idx >= len(item_ids) or item_ids[item_idx] != item_id:
            raise ValueError(f"Unknown item ID: {item_id}")
        neighbors = np.argsort(item_similarity[item_idx])[::-1]
        neighbors = neighbors[item_similarity[item_idx, neighbors] > 0][:n]
        result = pd.DataFrame({
            "item_id": item_ids[neighbors],
            "similarity": item_similarity[item_idx, neighbors],
            "shared_users": item_overlap[item_idx, neighbors].astype(int),
        })
        return result.merge(movies, on="item_id", how="left")


    example_movie = int(train.groupby("item_id").size().idxmax())
    print("Neighbors of:", movies.loc[movies.item_id == example_movie, "title"].iloc[0])
    display(nearest_items(example_movie))
    """,
)

add(
    "markdown",
    r"""
    The old seminar also used the number of **co-rating users** as an "intersection similarity." It is a useful support count, but not a normalized similarity: two very popular movies can have a large intersection even when their relative ratings disagree. Compare the top neighbors by the two criteria below.
    """,
)

add(
    "code",
    r"""
    example_idx = int(np.searchsorted(item_ids, example_movie))
    by_overlap = np.argsort(item_overlap[example_idx])[::-1]
    by_overlap = by_overlap[by_overlap != example_idx][:5]
    overlap_neighbors = pd.DataFrame({
        "item_id": item_ids[by_overlap],
        "shared_users": item_overlap[example_idx, by_overlap].astype(int),
        "adjusted_cosine": item_similarity[example_idx, by_overlap],
    }).merge(movies, on="item_id", how="left")
    display(overlap_neighbors)
    """,
)

add(
    "markdown",
    r"""
    ## 4. Predict ratings and compare methods

    For item CF, the neighbor evidence for target movie $i$ comes from movies the target user has rated. For user CF, it comes from similar users who rated $i$. In both cases we add a weighted average of neighbor residuals to the target user's training mean. We use at most 40 positive neighbors, return the user mean when none are available, and clip predictions to the valid 1–5 range. The fallback is also our baseline.

    **RMSE** penalizes large rating errors more than **MAE**. Both evaluate explicit rating prediction; they do not tell us whether users would click an unseen recommendation. For that, we would need a ranking protocol with carefully defined candidates and relevance.
    """,
)

add(
    "code",
    r"""
    K_NEIGHBORS = 40


    def weighted_prediction(similarities, neighbor_residuals, baseline, k=K_NEIGHBORS):
        positive = similarities > 0
        if not np.any(positive):
            return float(np.clip(baseline, 1, 5)), 0
        weights = similarities[positive]
        deviations = neighbor_residuals[positive]
        if len(weights) > k:
            strongest = np.argpartition(weights, -k)[-k:]
            weights, deviations = weights[strongest], deviations[strongest]
        estimate = baseline + np.dot(weights, deviations) / weights.sum()
        return float(np.clip(estimate, 1, 5)), len(weights)


    def predict_item(user_idx, item_idx, k=K_NEIGHBORS):
        history = centered.getrow(user_idx)
        return weighted_prediction(
            item_similarity[item_idx, history.indices],
            history.data,
            user_mean[user_idx],
            k,
        )


    def predict_user(user_idx, item_idx, k=K_NEIGHBORS):
        item_ratings = centered_csc.getcol(item_idx)
        return weighted_prediction(
            user_similarity[user_idx, item_ratings.indices],
            item_ratings.data,
            user_mean[user_idx],
            k,
        )


    # A recommendation must never use a rating hidden in the test set.
    sample_test = test.iloc[0]
    assert train_ratings[int(sample_test.user_idx), int(sample_test.item_idx)] == 0
    """,
)

add(
    "code",
    r"""
    truth = test["rating"].to_numpy(dtype=float)
    test_users = test["user_idx"].to_numpy(dtype=int)
    test_items = test["item_idx"].to_numpy(dtype=int)
    predictions = {
        "User-mean baseline": np.clip(user_mean[test_users], 1, 5),
    }
    support = {}
    for name, predictor in (("Item-to-item", predict_item), ("User-to-user", predict_user)):
        outputs = [predictor(user, item) for user, item in zip(test_users, test_items)]
        predictions[name] = np.array([value for value, _ in outputs])
        support[name] = np.array([count for _, count in outputs])

    rows = []
    for name, estimates in predictions.items():
        error = estimates - truth
        rows.append({
            "model": name,
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "MAE": np.mean(np.abs(error)),
            "neighbor_coverage": np.mean(support[name] > 0) if name in support else np.nan,
        })
    results = pd.DataFrame(rows).set_index("model")
    display(results.round(4))
    assert all(len(values) == len(truth) and np.isfinite(values).all() for values in predictions.values())
    """,
)

add(
    "markdown",
    r"""
    ## 5. Inspect recommendations

    The functions below rank items absent from the **training** history. Because we held out ratings, a hidden test item may appear in this list; this is expected for offline evaluation. A raw prediction of 5.0 based on one neighbor is fragile. We therefore require at least three positive neighbors and pull the ranking score toward the user's mean when support is low. This affects the displayed top-N ranking, not the RMSE/MAE calculation above.
    """,
)

add(
    "code",
    r"""
    def recommend(user_id, method="item", n=10, min_neighbors=3, confidence=5):
        user_idx = int(np.searchsorted(user_ids, user_id))
        if user_idx >= len(user_ids) or user_ids[user_idx] != user_id:
            raise ValueError(f"Unknown user ID: {user_id}")
        rated_items = set(train_ratings.getrow(user_idx).indices)
        predictor = predict_item if method == "item" else predict_user if method == "user" else None
        if predictor is None:
            raise ValueError("method must be 'item' or 'user'")
        candidates = []
        for item_idx in range(n_items):
            if item_idx in rated_items:
                continue
            estimate, neighbors = predictor(user_idx, item_idx)
            if neighbors >= min_neighbors:
                ranking_score = user_mean[user_idx] + (
                    (estimate - user_mean[user_idx]) * neighbors / (neighbors + confidence)
                )
                candidates.append((item_ids[item_idx], estimate, neighbors, ranking_score))
        result = pd.DataFrame(
            candidates,
            columns=["item_id", "estimated_rating", "neighbors", "ranking_score"],
        )
        result = result.sort_values(["ranking_score", "neighbors"], ascending=False).head(n)
        return result.merge(movies, on="item_id", how="left")


    EXAMPLE_USER_ID = int(user_ids[0])
    print("Item-based recommendations for user", EXAMPLE_USER_ID)
    display(recommend(EXAMPLE_USER_ID, "item"))
    print("User-based recommendations for the same user")
    display(recommend(EXAMPLE_USER_ID, "user"))
    seen_by_example_user = set(
        item_ids[train_ratings.getrow(int(np.searchsorted(user_ids, EXAMPLE_USER_ID))).indices]
    )
    assert not seen_by_example_user.intersection(recommend(EXAMPLE_USER_ID, "item")["item_id"])
    assert not seen_by_example_user.intersection(recommend(EXAMPLE_USER_ID, "user")["item_id"])
    """,
)

add(
    "markdown",
    r"""
    ## 6. What to explore next

    - **Neighborhood size and shrinkage:** try `K_NEIGHBORS` of 10 or 80, or `shrinkage` of 0 or 50 in `cosine_with_overlap`. Compare both error and neighbor coverage on the same held-out set. Tune on validation data; keep a separate final test set if you want an unbiased number.
    - **Popularity bias:** inspect whether the recommended movies are much more popular than a typical catalog item. Similarity based on co-ratings often favors popular objects.
    - **Cold start:** neither neighborhood model can infer preferences for a brand-new user or movie without interactions. A content-based or hybrid model can help.
    - **Ranking evaluation:** a rating RMSE win need not mean better top-10 recommendations. Define relevant held-out items, fix the same candidate pool for every model, and then calculate Recall@K or NDCG@K.
    """,
)

add(
    "code",
    r"""
    item_popularity = train.groupby("item_id").size().rename("train_ratings")
    print(f"Median movie popularity in training: {item_popularity.median():.0f} ratings")
    for method in ("item", "user"):
        recs = recommend(EXAMPLE_USER_ID, method, n=10)
        recommended_popularity = recs["item_id"].map(item_popularity).fillna(0)
        print(
            f"Median popularity among 10 {method}-CF recommendations: "
            f"{recommended_popularity.median():.0f} ratings"
        )
    print("This is an illustration for one user, not an aggregate bias estimate.")
    """,
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("seminar.ipynb")
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
