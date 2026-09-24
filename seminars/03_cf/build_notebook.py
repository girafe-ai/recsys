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

    **Learning goals:** distinguish item-based from user-based CF; understand centering, overlap and shrinkage; avoid evaluation leakage; interpret RMSE/MAE; inspect movie and user neighbors; compare our user CF with an open-source implementation.
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
            [sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib", "scikit-surprise==1.1.5"],
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
    ### EDA: насколько разрежены оценки и где находится длинный хвост?

    У соседских методов есть важная предпосылка: для двух пользователей или двух фильмов должно найтись достаточно общих оценок. Проверим плотность матрицы и долю оценок, приходящуюся на самые популярные фильмы. Эти числа сразу объясняют, почему `min_common` и способ обработки редких объектов важны.
    """,
)

add(
    "code",
    r"""
    movie_counts = ratings.groupby("item_id").size().sort_values(ascending=False)
    density = len(ratings) / (ratings.user_id.nunique() * ratings.item_id.nunique())
    top_tenth = max(1, int(np.ceil(0.1 * len(movie_counts))))
    top_tenth_share = movie_counts.iloc[:top_tenth].sum() / len(ratings)
    cumulative_share = movie_counts.cumsum().to_numpy() / movie_counts.sum()
    catalog_share = np.arange(1, len(movie_counts) + 1) / len(movie_counts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].plot(np.arange(1, len(movie_counts) + 1), movie_counts.to_numpy(), color="#278f88")
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
    ## 2. User-to-user и item-to-item: как строятся соседи

    [Вводная глава Яндекс Хендбука](https://education.yandex.ru/handbook/ml/article/intro-recsys) показывает две симметричные идеи коллаборативной фильтрации. Мы используем только матрицу оценок, без жанров и описаний фильмов.

    **User-to-user.** Строка матрицы — история пользователя. Для целевого пользователя находим других людей со схожими оценками общих фильмов. Чтобы предсказать оценку нового фильма, берём оценки этого фильма у соседей и поправляем их на личную среднюю оценку каждого соседа. Так пользователь, который всем фильмам ставит 5, не становится автоматически «щедрым» голосом для любого другого пользователя.

    **Item-to-item.** Столбец матрицы — оценки фильма разными людьми. Для нового фильма ищем похожие фильмы, уже оценённые целевым пользователем, и агрегируем отклонения этих оценок от его среднего. Сходство в этом подходе поведенческое: два фильма могут быть соседями, потому что их одинаково оценивает аудитория, даже если жанры различаются.

    В нашей реализации $x_{ui}=r_{ui}-\bar r_u$ для известных оценок и ноль для отсутствующих. Cosine similarity между строками $x_u$ даёт user-to-user, между столбцами — adjusted cosine для item-to-item:

    $$s(a,b)=\frac{x_a\cdot x_b}{\lVert x_a\rVert\lVert x_b\rVert}\cdot\frac{n_{ab}}{n_{ab}+\lambda}.$$

    $n_{ab}$ — число общих оценок, $\lambda$ ослабляет сходство по малому числу наблюдений. Мы обнуляем сходство при менее чем трёх общих оценках и используем для прогноза только положительных соседей. Это **cosine по центрированным векторам**; Pearson, посчитанный только на пересечении, может дать другое значение. Отсутствующая оценка остаётся неизвестной, а не отрицательной.

    **Когда методы затрудняются:** мало общих оценок, новый пользователь/фильм, сильный перекос популярности. На большом каталоге полные матрицы сходства, которые мы строим для наглядности, также становятся дорогими по памяти.
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
            "train_ratings": np.asarray(observed.sum(axis=0)).ravel()[neighbors].astype(int),
        })
        return result.merge(movies, on="item_id", how="left")[[
            "title", "item_id", "similarity", "shared_users", "train_ratings"
        ]]


    # Choose three recognizable films. Try another MovieLens item_id here.
    for example_movie in (50, 1, 483):  # Star Wars, Toy Story, Casablanca
        title = movies.loc[movies.item_id == example_movie, "title"].iloc[0]
        print(f"\nClosest movies to {title} (ID {example_movie}):")
        display(nearest_items(example_movie, n=7).round({"similarity": 3}))
    """,
)

add(
    "markdown",
    r"""
    **Как читать соседей по названиям.** У *Star Wars* среди ближайших обычно видны *Return of the Jedi* и *The Empire Strikes Back* — понятная проверка здравого смысла. У *Casablanca* появляются классические фильмы вроде *Citizen Kane* и *The Maltese Falcon*. У *Toy Story* список более смешанный: модель узнаёт совпадения во вкусах аудитории, а не жанр или сюжет. Смотрите также `shared_users`: высокая похожесть по нескольким людям менее надёжна, чем по сотням.
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
    ### Библиотечный user-to-user: Surprise

    [Surprise `KNNWithMeans`](https://surprise.readthedocs.io/en/stable/knn_inspired.html) реализует предсказание через похожих пользователей и поправку на их средние оценки. Мы задаём `user_based=True`, `k=40`, [Pearson baseline со shrinkage](https://surprise.readthedocs.io/en/stable/similarities.html) и минимум три общих фильма. Обычный Pearson может оказаться равным 1 даже по трём совпавшим оценкам; shrinkage ослабляет такие выводы. Обучаем библиотеку на **том же `train`**, затем проверяем на **том же `test`** — сравнивать RMSE иначе было бы некорректно.

    Результаты не обязаны совпадать с нашей реализацией: Surprise центрирует оценки относительно своих baseline-оценок и считает сходство по общим фильмам; наша модель использует cosine пользовательских отклонений и другую формулу shrinkage. По-разному устроен и fallback без достаточного числа соседей. Разница здесь — повод изучить определение сходства, а не признак ошибки сама по себе.
    """,
)

add(
    "code",
    r"""
    from surprise import Dataset, KNNWithMeans, Reader

    # String IDs make the raw/inner ID distinction explicit in Surprise.
    surprise_train = train[["user_id", "item_id", "rating"]].astype({
        "user_id": str, "item_id": str
    })
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(surprise_train, reader)
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
    surprise_testset = [
        (str(user), str(item), float(rating))
        for user, item, rating in test[["user_id", "item_id", "rating"]].itertuples(index=False, name=None)
    ]
    surprise_predictions = surprise_u2u.test(surprise_testset)
    library_estimates = np.array([prediction.est for prediction in surprise_predictions])
    library_support = np.array([
        prediction.details.get("actual_k", 0) for prediction in surprise_predictions
    ])
    assert len(library_estimates) == len(truth)
    assert np.isfinite(library_estimates).all()
    library_error = library_estimates - truth
    results.loc["Surprise user-to-user"] = {
        "RMSE": np.sqrt(np.mean(library_error ** 2)),
        "MAE": np.mean(np.abs(library_error)),
        "neighbor_coverage": np.mean(library_support >= 3),
    }
    display(results.round(4))
    """,
)

add(
    "markdown",
    r"""
    У user-to-user сосед — другой человек, поэтому одного `user_id` недостаточно для интуитивной проверки. Ниже для нескольких ближайших соседей показаны похожесть, число совместно оценённых фильмов и фильмы, которым **оба** поставили не меньше четырёх звёзд. Это иллюстрация на одном пользователе, не объяснение всех рекомендаций модели.
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
    title_by_id = movies.set_index("item_id")["title"].to_dict()
    reference_ratings = train.loc[train.user_id == example_user_id].set_index("item_id")["rating"]
    neighbor_rows = []
    for neighbor in positive_neighbors:
        neighbor_id = int(surprise_trainset.to_raw_uid(neighbor))
        neighbor_ratings = train.loc[train.user_id == neighbor_id].set_index("item_id")["rating"]
        common = reference_ratings.index.intersection(neighbor_ratings.index)
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
    display(pd.DataFrame(neighbor_rows).round({"shrunk_similarity": 3}))
    """,
)

add(
    "markdown",
    r"""
    ### EDA после оценки: где соседям не хватает данных?

    Разобьём скрытые оценки по числу **обучающих** оценок соответствующего фильма. Для редких фильмов соседей обычно меньше; это может ухудшать качество прогноза. Используем эту таблицу как диагностику на test, а не как набор для подбора `k` и `shrinkage`: подбирать гиперпараметры нужно на отдельной validation-выборке.
    """,
)

add(
    "code",
    r"""
    train_movie_count = train.groupby("item_id").size()
    popularity = test["item_id"].map(train_movie_count).to_numpy()
    popularity_bucket = pd.cut(
        popularity,
        bins=[0, 5, 20, 100, np.inf],
        labels=["1–5", "6–20", "21–100", "101+"],
    )
    diagnostics = pd.DataFrame({
        "popularity": popularity_bucket,
        "baseline_MAE": np.abs(predictions["User-mean baseline"] - truth),
        "item_MAE": np.abs(predictions["Item-to-item"] - truth),
        "user_MAE": np.abs(predictions["User-to-user"] - truth),
        "Surprise_MAE": np.abs(library_estimates - truth),
        "item_coverage": support["Item-to-item"] > 0,
        "user_coverage": support["User-to-user"] > 0,
    })
    by_popularity = diagnostics.groupby("popularity", observed=True).agg(
        n=("baseline_MAE", "size"),
        baseline_MAE=("baseline_MAE", "mean"),
        item_MAE=("item_MAE", "mean"),
        user_MAE=("user_MAE", "mean"),
        Surprise_MAE=("Surprise_MAE", "mean"),
        item_coverage=("item_coverage", "mean"),
        user_coverage=("user_coverage", "mean"),
    )
    display(by_popularity.round(3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    by_popularity[["baseline_MAE", "item_MAE", "user_MAE", "Surprise_MAE"]].rename(columns={
        "baseline_MAE": "User mean",
        "item_MAE": "Item CF",
        "user_MAE": "User CF",
        "Surprise_MAE": "Surprise U2U",
    }).plot.bar(ax=axes[0])
    axes[0].set(title="MAE by movie popularity", xlabel="Train ratings per movie", ylabel="MAE")
    by_popularity[["item_coverage", "user_coverage"]].rename(columns={
        "item_coverage": "Item CF",
        "user_coverage": "User CF",
    }).plot.line(ax=axes[1], marker="o")
    axes[1].set(title="Positive-neighbor coverage", xlabel="Train ratings per movie", ylabel="Fraction", ylim=(0, 1.05))
    axes[1].grid(alpha=0.2)
    plt.show()
    rare = by_popularity.loc["1–5"]
    print(
        f"Rare movies (1–5 train ratings): item-CF coverage {rare['item_coverage']:.1%}; "
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
    Для полноты сделаем top-10 тем же библиотечным user-to-user методом. Применяем такой же порог по числу соседей и то же осторожное ранжирование, что в собственной функции выше. Библиотека возвращает `actual_k` — реальное число положительных соседей, участвовавших в оценке.
    """,
)

add(
    "code",
    r"""
    def recommend_surprise_user(user_id, n=10, min_neighbors=3, confidence=5):
        user_idx = int(np.searchsorted(user_ids, user_id))
        if user_idx >= len(user_ids) or user_ids[user_idx] != user_id:
            raise ValueError(f"Unknown user ID: {user_id}")
        rated_items = set(train_ratings.getrow(user_idx).indices)
        candidates = []
        for item_idx, item_id in enumerate(item_ids):
            if item_idx in rated_items or item_idx not in known_items:
                continue
            prediction = surprise_u2u.predict(str(user_id), str(item_id))
            neighbors = prediction.details.get("actual_k", 0)
            if neighbors >= min_neighbors:
                ranking_score = user_mean[user_idx] + (
                    (prediction.est - user_mean[user_idx]) * neighbors / (neighbors + confidence)
                )
                candidates.append((item_id, prediction.est, neighbors, ranking_score))
        result = pd.DataFrame(
            candidates,
            columns=["item_id", "estimated_rating", "neighbors", "ranking_score"],
        )
        result = result.sort_values(["ranking_score", "neighbors"], ascending=False).head(n)
        return result.merge(movies, on="item_id", how="left")


    print("Surprise user-based recommendations for user", EXAMPLE_USER_ID)
    display(recommend_surprise_user(EXAMPLE_USER_ID))
    assert not seen_by_example_user.intersection(
        recommend_surprise_user(EXAMPLE_USER_ID)["item_id"]
    )
    """,
)

add(
    "markdown",
    r"""
    ## 6. What to explore next

    - **Neighborhood size and shrinkage:** try `K_NEIGHBORS` of 10 or 80, or `shrinkage` of 0 or 50 in `cosine_with_overlap`. Compare both error and neighbor coverage on the same held-out set. Tune on validation data; keep a separate final test set if you want an unbiased number.
    - **Popularity and coverage:** compare top-N popularity for item CF, user CF and Surprise across many users. Our one-user example may differ from the aggregate. Also ask what fraction of catalog items each method can recommend.
    - **Who benefits:** group users by training-history length and compare their errors. Sparse histories often produce less reliable neighbors.
    - **Similarity definitions:** compare the current centered cosine with Surprise Pearson and different overlap thresholds on a validation split.
    - **Cold start:** neither neighborhood model can infer preferences for a brand-new user or movie without interactions. A content-based or hybrid model can help.
    - **Ranking evaluation:** a rating RMSE win need not mean better top-10 recommendations. Define relevant held-out items, fix the same candidate pool for every model, and then calculate Recall@K or NDCG@K.
    """,
)

add(
    "code",
    r"""
    item_popularity = train.groupby("item_id").size().rename("train_ratings")
    print(f"Median movie popularity in training: {item_popularity.median():.0f} ratings")
    for method in ("item", "user", "Surprise user"):
        recs = (
            recommend_surprise_user(EXAMPLE_USER_ID, n=10)
            if method == "Surprise user"
            else recommend(EXAMPLE_USER_ID, method, n=10)
        )
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
