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

    **Learning goals:** distinguish item-based from user-based CF; understand centering, overlap and shrinkage; avoid evaluation leakage; interpret RMSE/MAE; inspect movie and user neighbors; compare both approaches with open-source implementations. All tabular work uses Polars.
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
            [sys.executable, "-m", "pip", "install", "-q", "numpy", "polars", "scipy", "matplotlib", "scikit-surprise==1.1.5"],
            check=True,
        )

    from io import BytesIO
    from pathlib import Path
    from urllib.request import urlopen
    from zipfile import ZipFile

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
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
    print("Data cache is ready.")
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

    ratings = pl.read_csv(
        ratings_path,
        separator="\t",
        has_header=False,
        new_columns=["user_id", "item_id", "rating", "timestamp"],
    )
    movies = pl.read_csv(
        movies_path,
        separator="|",
        encoding="iso-8859-1",
        has_header=False,
        columns=[0, 1],
    ).rename({"column_1": "item_id", "column_2": "title"})
    assert ratings.height == 100_000
    assert ratings.select(pl.col("rating").is_between(1, 5).all()).item()
    assert ratings.group_by(["user_id", "item_id"]).len().filter(pl.col("len") > 1).is_empty()
    display(ratings.head())
    print(f"{ratings['user_id'].n_unique()} users, {ratings['item_id'].n_unique()} rated items")
    """,
)

add(
    "code",
    r"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout="constrained")
    rating_counts = ratings.group_by("rating").len().sort("rating")
    axes[0].bar(rating_counts["rating"].to_numpy(), rating_counts["len"].to_numpy(), color="#278f88")
    axes[0].set(title="Rating distribution", xlabel="Stars", ylabel="Interactions")
    user_activity = ratings.group_by("user_id").len()["len"].to_numpy()
    axes[1].hist(user_activity, bins=35, color="#278f88")
    axes[1].set(title="Ratings per user", xlabel="Count", ylabel="Users")
    item_activity = ratings.group_by("item_id").len()["len"].to_numpy()
    axes[2].hist(item_activity, bins=35, color="#278f88")
    axes[2].set(title="Ratings per item", xlabel="Count", ylabel="Items")
    plt.show()
    """,
)

add(
    "markdown",
    r"""
    ### EDA: sparsity and the long tail

    Neighborhood methods need enough co-ratings between two users or two movies. Matrix density and the share of ratings assigned to popular movies help explain why `min_common` and rare-item handling matter.
    """,
)

add(
    "code",
    r"""
    movie_counts = ratings.group_by("item_id").len().sort("len", descending=True)["len"].to_numpy()
    density = ratings.height / (ratings["user_id"].n_unique() * ratings["item_id"].n_unique())
    top_tenth = max(1, int(np.ceil(0.1 * len(movie_counts))))
    top_tenth_share = movie_counts[:top_tenth].sum() / ratings.height
    cumulative_share = movie_counts.cumsum() / movie_counts.sum()
    catalog_share = np.arange(1, len(movie_counts) + 1) / len(movie_counts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(np.arange(1, len(movie_counts) + 1), movie_counts, color="#278f88")
    axes[0].set(yscale="log", xlabel="Movie popularity rank", ylabel="Number of ratings (log)", title="Long tail of movies")
    axes[1].plot(catalog_share, cumulative_share, color="#278f88", label="MovieLens 100K")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Even distribution")
    axes[1].set(xlabel="Fraction of most popular movies", ylabel="Fraction of all ratings", title="Where ratings concentrate")
    axes[1].legend()
    plt.show()
    print(f"Observed matrix density: {density:.1%} (missing values are unknown)")
    print(f"Top 10% of movies receive {top_tenth_share:.1%} of all ratings")
    """,
)

add(
    "markdown",
    r"""
    ## 2. User-to-user and item-to-item: building neighborhoods

    The [Yandex ML Handbook introduction](https://education.yandex.ru/handbook/ml/article/intro-recsys) presents two symmetric collaborative-filtering ideas. We use only the rating matrix, not movie genres or descriptions.

    **User-to-user.** Each matrix row is a user's history. We find people who rate the same movies similarly, then predict a new rating from those neighbors' ratings after adjusting for each neighbor's typical rating level. A generous rater who gives everything five stars does not automatically dominate.

    **Item-to-item.** Each column represents one movie's ratings across people. We find movies similar to the target movie in the user's history and aggregate how far their ratings lie above or below that user's mean. This similarity is behavioral: different genres can still attract similar rating patterns.

    Our implementation uses $x_{ui}=r_{ui}-\bar r_u$ for observed ratings and zero for missing entries. Cosine similarity between rows gives user-to-user; between columns, adjusted cosine for item-to-item:

    $$s(a,b)=\frac{x_a\cdot x_b}{\lVert x_a\rVert\lVert x_b\rVert}\cdot\frac{n_{ab}}{n_{ab}+\lambda}.$$

    Here $n_{ab}$ is the overlap count and $\lambda$ shrinks similarities supported by little data. We discard pairs with fewer than three co-ratings and use only positive neighbors for prediction. This is **cosine on centered vectors**; Pearson computed on the overlap can differ. A missing rating remains unknown, not negative.

    **Where these methods struggle:** scarce overlap, new users or movies, and popularity imbalance. Full similarity matrices are useful for teaching but become expensive on a large catalog.
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
    user_ids = np.sort(ratings["user_id"].unique().to_numpy())
    item_ids = np.sort(movies["item_id"].unique().to_numpy())
    ratings = ratings.with_columns(
        pl.Series("user_idx", np.searchsorted(user_ids, ratings["user_id"].to_numpy())),
        pl.Series("item_idx", np.searchsorted(item_ids, ratings["item_id"].to_numpy())),
    ).with_row_index("row_nr")

    rng = np.random.default_rng(RANDOM_STATE)
    test_mask = np.zeros(ratings.height, dtype=bool)
    for indices in ratings.group_by("user_idx").agg(pl.col("row_nr")).sort("user_idx")["row_nr"]:
        indices = indices.to_numpy()
        holdout_count = min(max(1, round(0.2 * len(indices))), len(indices) - 1)
        test_mask[rng.choice(indices, size=holdout_count, replace=False)] = True

    train = ratings.filter(~pl.Series(test_mask))
    test_all = ratings.filter(pl.Series(test_mask))
    known_items = set(train["item_idx"].to_list())
    test = test_all.filter(pl.col("item_idx").is_in(known_items))
    assert set(train["row_nr"].to_list()).isdisjoint(test_all["row_nr"].to_list())
    assert train["user_idx"].n_unique() == len(user_ids)
    print(f"Train: {train.height:,}; held out: {test_all.height:,}; evaluated: {test.height:,}")
    print(f"Cold-item held-out rows excluded: {test_all.height - test.height}")
    """,
)

add(
    "code",
    r"""
    n_users, n_items = len(user_ids), len(item_ids)
    user_count = np.bincount(train["user_idx"].to_numpy(), minlength=n_users)
    user_sum = np.bincount(
        train["user_idx"].to_numpy(), weights=train["rating"].to_numpy(), minlength=n_users
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
    values = train["rating"].to_numpy().astype(float)
    residuals = values - user_mean[rows]
    shape = (n_users, n_items)
    train_ratings = csr_matrix((values, (rows, cols)), shape=shape)
    centered = csr_matrix((residuals, (rows, cols)), shape=shape)
    centered_csc = centered.tocsc()
    observed = csr_matrix((np.ones(train.height), (rows, cols)), shape=shape)
    assert train_ratings.nnz == train.height
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
        result = pl.DataFrame({
            "item_id": item_ids[neighbors],
            "similarity": item_similarity[item_idx, neighbors],
            "shared_users": item_overlap[item_idx, neighbors].astype(int),
            "train_ratings": np.asarray(observed.sum(axis=0)).ravel()[neighbors].astype(int),
        })
        return result.join(movies, on="item_id", how="left").select(
            "title", "item_id", "similarity", "shared_users", "train_ratings"
        )


    # Choose three recognizable films. Try another MovieLens item_id here.
    for example_movie in (50, 1, 483):  # Star Wars, Toy Story, Casablanca
        title = movies.filter(pl.col("item_id") == example_movie)["title"].item()
        print(f"\nClosest movies to {title} (ID {example_movie}):")
        display(nearest_items(example_movie, n=7).with_columns(pl.col("similarity").round(3)))
    """,
)

add(
    "markdown",
    r"""
    **Interpreting movie neighbors by title.** *Star Wars* tends to be close to *Return of the Jedi* and *The Empire Strikes Back*, a useful sanity check. *Casablanca* can have classic-film neighbors such as *Citizen Kane* and *The Maltese Falcon*. *Toy Story* may have a more varied list: the model detects audience behavior, not plot or genre. Also inspect `shared_users`: a large similarity based on a few people is less convincing than one supported by hundreds.
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
    example_movie = 50  # Star Wars, to compare the two neighbor definitions
    example_idx = int(np.searchsorted(item_ids, example_movie))
    by_overlap = np.argsort(item_overlap[example_idx])[::-1]
    by_overlap = by_overlap[by_overlap != example_idx][:5]
    overlap_neighbors = pl.DataFrame({
        "item_id": item_ids[by_overlap],
        "shared_users": item_overlap[example_idx, by_overlap].astype(int),
        "adjusted_cosine": item_similarity[example_idx, by_overlap],
    }).join(movies, on="item_id", how="left")
    display(overlap_neighbors)
    """,
)

add(
    "markdown",
    r"""
    ## 4. Predict ratings and compare methods

    For item CF, the neighbor evidence for target movie $i$ comes from movies the target user has rated. For user CF, it comes from similar users who rated $i$. The weighted prediction is:

    $$\hat r_{ui}^{\text{item}}=\bar r_u+\frac{\sum_{j\in N_i(u)}s(i,j)(r_{uj}-\bar r_u)}{\sum_{j\in N_i(u)}s(i,j)},$$
    $$\hat r_{ui}^{\text{user}}=\bar r_u+\frac{\sum_{v\in N_u(i)}s(u,v)(r_{vi}-\bar r_v)}{\sum_{v\in N_u(i)}s(u,v)}.$$

    We use at most 40 positive neighbors, return the user mean when none are available, and clip predictions to the valid 1–5 range. The fallback is also our baseline.

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
    sample_test = test.row(0, named=True)
    assert train_ratings[int(sample_test["user_idx"]), int(sample_test["item_idx"])] == 0
    """,
)

add(
    "code",
    r"""
    truth = test["rating"].to_numpy().astype(float)
    test_users = test["user_idx"].to_numpy().astype(int)
    test_items = test["item_idx"].to_numpy().astype(int)
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
    results = pl.DataFrame(rows)
    display(results.with_columns(pl.selectors.numeric().round(4)))
    assert all(len(values) == len(truth) and np.isfinite(values).all() for values in predictions.values())
    """,
)

add(
    "markdown",
    r"""
    ### Open-source neighborhood models: Surprise

    [Surprise `KNNWithMeans`](https://surprise.readthedocs.io/en/stable/knn_inspired.html) predicts with similar users or movies and an adjustment for their mean ratings. We fit both `user_based=True` and `user_based=False` with `k=40`, [Pearson baseline with shrinkage](https://surprise.readthedocs.io/en/stable/similarities.html), and a minimum of three common ratings. Ordinary Pearson can equal one even on just three co-ratings; shrinkage reduces that overconfidence. Both library models use **the same `train`** and are evaluated on **the same `test`** as ours.

    The results need not match: Surprise's Pearson-baseline similarity and mean-adjusted prediction differ from our globally centered cosine, overlap shrinkage, positive-neighbor selection, and fallback. A difference invites inspection of the exact model definitions; it is not automatically a bug.
    """,
)

add(
    "code",
    r"""
    from surprise import Dataset, KNNWithMeans, Reader

    # File input keeps all tabular work in Polars; Surprise parses its own TSV.
    surprise_train_path = DATA_DIR / "surprise_train.tsv"
    train.select("user_id", "item_id", "rating").write_csv(
        surprise_train_path, separator="\t", include_header=False
    )
    reader = Reader(line_format="user item rating", sep="\t", rating_scale=(1, 5))
    surprise_data = Dataset.load_from_file(str(surprise_train_path), reader)
    surprise_trainset = surprise_data.build_full_trainset()
    surprise_u2u = KNNWithMeans(
        k=K_NEIGHBORS,
        min_k=3,
        sim_options={
            "name": "pearson_baseline",
            "user_based": True,
            "min_support": 3,
            "shrinkage": 20,
        },
        verbose=False,
    )
    surprise_u2u.fit(surprise_trainset)
    surprise_i2i = KNNWithMeans(
        k=K_NEIGHBORS,
        min_k=3,
        sim_options={
            "name": "pearson_baseline",
            "user_based": False,
            "min_support": 3,
            "shrinkage": 20,
        },
        verbose=False,
    )
    surprise_i2i.fit(surprise_trainset)
    surprise_testset = [
        (str(user), str(item), float(rating))
        for user, item, rating in test.select("user_id", "item_id", "rating").iter_rows()
    ]
    library_estimates = {}
    library_support = {}
    for name, model in (("Surprise user-to-user", surprise_u2u), ("Surprise item-to-item", surprise_i2i)):
        model_predictions = model.test(surprise_testset)
        estimates = np.array([prediction.est for prediction in model_predictions])
        actual_k = np.array([prediction.details.get("actual_k", 0) for prediction in model_predictions])
        assert len(estimates) == len(truth) and np.isfinite(estimates).all()
        library_estimates[name] = estimates
        library_support[name] = actual_k
        error = estimates - truth
        results = pl.concat([results, pl.DataFrame([{
            "model": name,
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "MAE": np.mean(np.abs(error)),
            "neighbor_coverage": np.mean(actual_k >= 3),
        }])])
    display(results.with_columns(pl.selectors.numeric().round(4)))
    """,
)

add(
    "markdown",
    r"""
    For user-to-user, a neighbor is another person, so a `user_id` alone is not an intuitive check. Below we show similarity, number of co-rated movies, and movies **both** people rated at least four stars for a few neighbors. This is one example, not a complete explanation of the model's recommendations.
    """,
)

add(
    "code",
    r"""
    example_user_id = 1
    inner_user = surprise_trainset.to_inner_uid(str(example_user_id))
    raw_neighbors = surprise_u2u.get_neighbors(inner_user, k=20)
    positive_neighbors = [
        neighbor for neighbor in raw_neighbors
        if surprise_u2u.sim[inner_user, neighbor] > 0
    ][:5]
    title_by_id = dict(movies.select("item_id", "title").iter_rows())
    reference_ratings = dict(train.filter(pl.col("user_id") == example_user_id).select("item_id", "rating").iter_rows())
    neighbor_rows = []
    for neighbor in positive_neighbors:
        neighbor_id = int(surprise_trainset.to_raw_uid(neighbor))
        neighbor_ratings = dict(train.filter(pl.col("user_id") == neighbor_id).select("item_id", "rating").iter_rows())
        common = reference_ratings.keys() & neighbor_ratings.keys()
        liked_together = [
            title_by_id[item_id] for item_id in common
            if reference_ratings[item_id] >= 4 and neighbor_ratings[item_id] >= 4
        ]
        neighbor_rows.append({
            "neighbor_user_id": neighbor_id,
            "shrunk_similarity": surprise_u2u.sim[inner_user, neighbor],
            "shared_movies": len(common),
            "both_liked": "; ".join(liked_together[:3]) or "—",
        })
    display(pl.DataFrame(neighbor_rows).with_columns(pl.col("shrunk_similarity").round(3)))
    """,
)

add(
    "markdown",
    r"""
    ### EDA after evaluation: where do neighborhoods lack evidence?

    Group held-out ratings by the number of **training** ratings for their movies. Rare movies tend to have fewer useful neighbors. This test-set table is a diagnostic, not a basis for tuning `k` or shrinkage; hyperparameter selection needs a separate validation set.
    """,
)

add(
    "code",
    r"""
    train_movie_count = dict(train.group_by("item_id").len().iter_rows())
    popularity = np.array([train_movie_count[item_id] for item_id in test["item_id"].to_list()])
    diagnostics = pl.DataFrame({
        "train_ratings": popularity,
        "baseline_MAE": np.abs(predictions["User-mean baseline"] - truth),
        "item_MAE": np.abs(predictions["Item-to-item"] - truth),
        "user_MAE": np.abs(predictions["User-to-user"] - truth),
        "Surprise_U2U_MAE": np.abs(library_estimates["Surprise user-to-user"] - truth),
        "Surprise_I2I_MAE": np.abs(library_estimates["Surprise item-to-item"] - truth),
        "item_coverage": support["Item-to-item"] > 0,
        "user_coverage": support["User-to-user"] > 0,
    }).with_columns(
        pl.when(pl.col("train_ratings") <= 5).then(pl.lit("1-5"))
        .when(pl.col("train_ratings") <= 20).then(pl.lit("6-20"))
        .when(pl.col("train_ratings") <= 100).then(pl.lit("21-100"))
        .otherwise(pl.lit("101+"))
        .alias("popularity"),
        pl.when(pl.col("train_ratings") <= 5).then(0)
        .when(pl.col("train_ratings") <= 20).then(1)
        .when(pl.col("train_ratings") <= 100).then(2)
        .otherwise(3)
        .alias("bucket_order"),
    )
    by_popularity = diagnostics.group_by("bucket_order", "popularity").agg(
        pl.len().alias("n"),
        *[pl.col(column).mean() for column in (
            "baseline_MAE", "item_MAE", "user_MAE", "Surprise_U2U_MAE", "Surprise_I2I_MAE",
            "item_coverage", "user_coverage",
        )],
    ).sort("bucket_order")
    display(by_popularity.drop("bucket_order").with_columns(pl.selectors.numeric().round(3)))

    labels = by_popularity["popularity"].to_list()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout="constrained")
    for offset, (column, label) in enumerate((
        ("baseline_MAE", "User mean"),
        ("item_MAE", "Item CF"),
        ("user_MAE", "User CF"),
        ("Surprise_U2U_MAE", "Surprise U2U"),
        ("Surprise_I2I_MAE", "Surprise I2I"),
    )):
        axes[0].bar(x + (offset - 2) * 0.16, by_popularity[column].to_numpy(), width=0.16, label=label)
    axes[0].set(xticks=x, xticklabels=labels, title="MAE by movie popularity", xlabel="Train ratings per movie", ylabel="MAE")
    axes[0].legend(fontsize=8)
    for column, label in (("item_coverage", "Item CF"), ("user_coverage", "User CF")):
        axes[1].plot(x, by_popularity[column].to_numpy(), marker="o", label=label)
    axes[1].set(xticks=x, xticklabels=labels, title="Positive-neighbor coverage", xlabel="Train ratings per movie", ylabel="Fraction", ylim=(0, 1.05))
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    plt.show()
    rare = by_popularity.filter(pl.col("popularity") == "1-5").row(0, named=True)
    print(
        f"Rare movies (1-5 train ratings): item-CF coverage {rare['item_coverage']:.1%}; "
        f"MAE {rare['item_MAE']:.3f} versus user-mean baseline {rare['baseline_MAE']:.3f}."
    )
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
        result = pl.DataFrame(
            candidates,
            schema=["item_id", "estimated_rating", "neighbors", "ranking_score"],
            orient="row",
        )
        result = result.sort(["ranking_score", "neighbors"], descending=[True, True]).head(n)
        return result.join(movies, on="item_id", how="left")


    EXAMPLE_USER_ID = int(user_ids[0])
    print("Item-based recommendations for user", EXAMPLE_USER_ID)
    display(recommend(EXAMPLE_USER_ID, "item"))
    print("User-based recommendations for the same user")
    display(recommend(EXAMPLE_USER_ID, "user"))
    seen_by_example_user = set(
        item_ids[train_ratings.getrow(int(np.searchsorted(user_ids, EXAMPLE_USER_ID))).indices]
    )
    assert not seen_by_example_user.intersection(recommend(EXAMPLE_USER_ID, "item")["item_id"].to_list())
    assert not seen_by_example_user.intersection(recommend(EXAMPLE_USER_ID, "user")["item_id"].to_list())
    """,
)

add(
    "markdown",
    r"""
    For completeness, generate top-10 recommendations with the open-source user-to-user model. We apply the same minimum-neighbor threshold and cautious ranking adjustment as above. Surprise reports `actual_k`, the number of neighbors actually used in a prediction.
    """,
)

add(
    "code",
    r"""
    def recommend_surprise(user_id, model, n=10, min_neighbors=3, confidence=5):
        user_idx = int(np.searchsorted(user_ids, user_id))
        if user_idx >= len(user_ids) or user_ids[user_idx] != user_id:
            raise ValueError(f"Unknown user ID: {user_id}")
        rated_items = set(train_ratings.getrow(user_idx).indices)
        candidates = []
        for item_idx, item_id in enumerate(item_ids):
            if item_idx in rated_items or item_idx not in known_items:
                continue
            prediction = model.predict(str(user_id), str(item_id))
            neighbors = prediction.details.get("actual_k", 0)
            if neighbors >= min_neighbors:
                ranking_score = user_mean[user_idx] + (
                    (prediction.est - user_mean[user_idx]) * neighbors / (neighbors + confidence)
                )
                candidates.append((item_id, prediction.est, neighbors, ranking_score))
        result = pl.DataFrame(
            candidates,
            schema=["item_id", "estimated_rating", "neighbors", "ranking_score"],
            orient="row",
        )
        result = result.sort(["ranking_score", "neighbors"], descending=[True, True]).head(n)
        return result.join(movies, on="item_id", how="left")


    print("Surprise user-based recommendations for user", EXAMPLE_USER_ID)
    display(recommend_surprise(EXAMPLE_USER_ID, surprise_u2u))
    assert not seen_by_example_user.intersection(
        recommend_surprise(EXAMPLE_USER_ID, surprise_u2u)["item_id"].to_list()
    )
    """,
)

add(
    "markdown",
    r"""
    ### Our item-to-item vs Surprise item-to-item: recommendation overlap

    We now compare two *item-based* recommenders on exactly the same training history and candidate universe: movies that appear in training and that the user has not rated there. Both use up to 40 neighbors, require at least three usable neighbors, and apply the same confidence adjustment before ranking. The table above also reports their RMSE and MAE on the identical held-out ratings.

    The top-10 overlap is $|A_{10}\cap B_{10}|/10$; Jaccard is $|A_{10}\cap B_{10}|/|A_{10}\cup B_{10}|$. These measure *agreement between models*, not recommendation quality. A low overlap does not by itself tell us which model is better.
    """,
)

add(
    "code",
    r"""
    comparison_users = (1, 10, 50)
    overlap_rows = []
    for comparison_user in comparison_users:
        ours = recommend(comparison_user, "item", n=10)
        library = recommend_surprise(comparison_user, surprise_i2i, n=10)
        ours_ids = set(ours["item_id"].to_list())
        library_ids = set(library["item_id"].to_list())
        user_idx = int(np.searchsorted(user_ids, comparison_user))
        seen = set(train_ratings.getrow(user_idx).indices)
        candidate_ids = {int(item_ids[item_idx]) for item_idx in known_items - seen}
        assert ours_ids <= candidate_ids and library_ids <= candidate_ids
        assert len(ours_ids) == len(library_ids) == 10
        common_ids = ours_ids & library_ids
        overlap_rows.append({
            "user_id": comparison_user,
            "our_top_n": len(ours_ids),
            "library_top_n": len(library_ids),
            "common_movies": len(common_ids),
            "overlap_at_10": len(common_ids) / 10,
            "jaccard_at_10": len(common_ids) / len(ours_ids | library_ids),
        })
    overlap_table = pl.DataFrame(overlap_rows)
    display(overlap_table.with_columns(pl.selectors.float().round(3)))
    print(f"Mean overlap@10 across these users: {overlap_table['overlap_at_10'].mean():.1%}")

    our_top = recommend(EXAMPLE_USER_ID, "item", n=10)
    library_top = recommend_surprise(EXAMPLE_USER_ID, surprise_i2i, n=10)
    title_by_id = dict(movies.select("item_id", "title").iter_rows())
    comparison = pl.DataFrame({
        "rank": range(1, 11),
        "our_item_id": our_top["item_id"],
        "our_movie": [title_by_id[item_id] for item_id in our_top["item_id"]],
        "library_item_id": library_top["item_id"],
        "library_movie": [title_by_id[item_id] for item_id in library_top["item_id"]],
    })
    display(comparison)
    shared_titles = [title_by_id[item_id] for item_id in our_top["item_id"] if item_id in set(library_top["item_id"].to_list())]
    print("Movies in both top-10 lists:", "; ".join(shared_titles) if shared_titles else "none")
    """,
)

add(
    "markdown",
    r"""
    **Why might the lists differ?** Our item similarity is cosine on *user-centered* ratings, with an explicit overlap factor $n/(n+10)$ and zero similarity below three co-ratings. Surprise uses Pearson-baseline similarity computed on co-raters with its own shrinkage, then `KNNWithMeans` adjusts predictions using *item means* rather than our user's mean. Positive-neighbor selection, `min_k` fallback, clipping, and the number of eligible neighbors can therefore differ even with the same `k`. The confidence adjustment above is identical, but it acts on different predictions and `actual_k` values. Small score differences can reorder many movies near rank 10. Compare the shared titles and rank positions; overlap is not a substitute for held-out ranking metrics or user feedback.
    """,
)

add(
    "markdown",
    r"""
    ## 6. What to explore next

    - **Neighborhood size and shrinkage:** try `K_NEIGHBORS` of 10 or 80, or `shrinkage` of 0 or 50 in `cosine_with_overlap`. Compare both error and neighbor coverage on the same held-out set. Tune on validation data; keep a separate final test set if you want an unbiased number.
    - **Popularity and coverage:** compare top-N popularity for both item and user CF implementations across many users. Our few-user examples may differ from the aggregate. Also ask what fraction of catalog items each method can recommend.
    - **Who benefits:** group users by training-history length and compare their errors. Sparse histories often produce less reliable neighbors.
    - **Similarity definitions:** compare the current centered cosine with Surprise Pearson and different overlap thresholds on a validation split.
    - **Cold start:** neither neighborhood model can infer preferences for a brand-new user or movie without interactions. A content-based or hybrid model can help.
    - **Ranking evaluation:** a rating RMSE win need not mean better top-10 recommendations. Define relevant held-out items, fix the same candidate pool for every model, and then calculate Recall@K or NDCG@K.
    """,
)

add(
    "code",
    r"""
    item_popularity = dict(train.group_by("item_id").len().iter_rows())
    print(f"Median movie popularity in training: {np.median(list(item_popularity.values())):.0f} ratings")
    for method in ("Our item", "Our user", "Surprise item", "Surprise user"):
        if method == "Surprise item":
            recs = recommend_surprise(EXAMPLE_USER_ID, surprise_i2i, n=10)
        elif method == "Surprise user":
            recs = recommend_surprise(EXAMPLE_USER_ID, surprise_u2u, n=10)
        else:
            recs = recommend(EXAMPLE_USER_ID, "item" if method == "Our item" else "user", n=10)
        recommended_popularity = [item_popularity.get(item_id, 0) for item_id in recs["item_id"].to_list()]
        print(
            f"Median popularity among 10 {method}-CF recommendations: "
            f"{np.median(recommended_popularity):.0f} ratings"
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
