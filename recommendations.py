from math import isnan, sqrt

from scipy.stats import pearsonr


def similar_films(critics_dict, person1, person2):
    sim_film = []

    for film in critics_dict[person1]:
        if film in critics_dict[person2]:
            sim_film.append(film)

    return sim_film


def sim_distance(critics_dict, person1, person2):
    sim_films = similar_films(critics_dict, person1, person2)

    if len(sim_films) == 0:
        return 0

    sum_of_euclead_dist = 0

    for film in sim_films:
        sum_of_euclead_dist += pow(
            critics_dict[person1][film] - critics_dict[person2][film], 2
        )

    return 1 / (1 + sum_of_euclead_dist)


def sim_pearson(critics_dict, person1, person2):
    sim_films = similar_films(critics_dict, person1, person2)

    if len(sim_films) < 2:
        return 0

    scores1 = []
    scores2 = []
    for film in sim_films:
        scores1.append(critics_dict[person1][film])
        scores2.append(critics_dict[person2][film])

    r = pearsonr(scores1, scores2)[0]

    if isnan(r):
        r = 0
    return r


def top_matches(critics_dict, person, n=5, similarity=sim_pearson):
    scores = []

    for other in critics_dict:
        if other != person:
            scores.append((similarity(critics_dict, person, other), other))

    scores.sort(reverse=True)

    return scores[0:n]


def get_recommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    sim_sums = {}

    for other in prefs:
        if other == person:
            continue

        sim = similarity(prefs, person, other)

        if sim <= 0:
            continue

        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] = totals[item] + prefs[other][item] * sim

                sim_sums.setdefault(item, 0)
                sim_sums[item] = sim_sums[item] + sim
    rankings = [(total / sim_sums[item], item) for item, total in totals.items()]

    rankings.sort(reverse=True)
    return rankings


def transform_prefs(critics_dict):
    result = {}

    for person in critics_dict:
        for item in critics_dict[person]:
            result.setdefault(item, {})
            result[item][person] = critics_dict[person][item]

    return result


def calculate_similar_items(prefs, n=10):
    result = {}
    item_prefs = transform_prefs(prefs)

    c = 0
    for item in item_prefs:
        c = c + 1
        if c % 100 == 0:
            print("%d / %d" % (c, len(item_prefs)))

        scores = top_matches(item_prefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


def get_recommended_items(prefs, item_match, user):
    user_ratings = prefs[user]
    scores = {}
    total_sim = {}

    for (item, rating) in user_ratings.items():
        for (similarity, item2) in item_match[item]:
            if item2 in user_ratings:
                continue

            scores.setdefault(item2, 0)
            scores[item2] = scores[item2] + similarity * rating

            total_sim.setdefault(item2, 0)
            total_sim[item2] = total_sim[item2] + similarity
            if total_sim[item2] == 0:
                total_sim[item2] = 0.0000001  # чтобы избежать деления на ноль

    rankings = [(score / total_sim[item], item) for item, score in scores.items()]

    rankings.sort(reverse=True)
    return rankings


def load_movie_lens():
    movies = {}
    for line in open("u.item", encoding="ISO-8859-1"):
        (id, title) = line.split("|")[0:2]
        movies[id] = title

    prefs = {}
    for line in open("u.data"):
        (user, movieid, rating, ts) = line.split("\t")
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
    return prefs
