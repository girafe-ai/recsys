import pandas as pd
import numpy as np


def dcg_score(y_true, y_score, k=10, gains="linear"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="linear"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def calculate_grouped_ndcg_with_embeddings(valDf, model, k):
    data = valDf.copy()
    
    user_vecs = model.user_factors
    item_vecs = model.item_factors
    
    data["predicted_rating"] = data.apply(lambda row: user_vecs[row["user_id"], :].dot(item_vecs[row["movie_id"], :].T), axis=1)
    
    nonnull_users = set(data[data.rating > 0].user_id)
    data = data[data.user_id.isin(nonnull_users)]
    
    return np.mean(data.groupby("user_id").apply(lambda x: ndcg_score(x.rating, x.predicted_rating, k)))

def calculate_grouped_ndcg_random(trainDf, valDf, k, i):
    data = valDf.copy()
    
    videos_popularity = trainDf.groupby(["movie_id"], as_index=False)["rating"].sum()
    videos_popularity.rename(columns={"rating": "predicted_rating"}, inplace=True)

    data = data.merge(videos_popularity, on=["movie_id"], how='left')
    
    nonnull_users = set(data[data.rating > 0].user_id)
    data = data[data.user_id.isin(nonnull_users)]

    np.random.seed(i)
    return np.mean(data.groupby("user_id").apply(lambda x: ndcg_score(x.rating, np.random.permutation(x.predicted_rating), k)))

def calculate_grouped_ndcg_sum_popularity(trainDf, valDf, k):
    data = valDf.copy()
    
    videos_popularity = trainDf.groupby(["movie_id"], as_index=False)["rating"].sum()
    videos_popularity.rename(columns={"rating": "predicted_rating"}, inplace=True)

    data = data.merge(videos_popularity, on=["movie_id"], how='left')
    
    nonnull_users = set(data[data.rating > 0].user_id)
    data = data[data.user_id.isin(nonnull_users)]

    return np.mean(data.groupby("user_id").apply(lambda x: ndcg_score(x.rating, x.predicted_rating, k)))

def calculate_grouped_ndcg_for_bert4rec_output(valDf, model_scores, k):
    data = valDf.copy()
    
    data["predicted_rating"] = data.apply(lambda row: model_scores[row["user_id"]][row["movie_id"]], axis=1)
    
    nonnull_users = set(data[data.rating > 0].user_id)
    data = data[data.user_id.isin(nonnull_users)]
    
    return np.mean(data.groupby("user_id").apply(lambda x: ndcg_score(x.rating, x.predicted_rating, k)))