import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import gdown
import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from qdrant_client import QdrantClient, models


def download_data(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".h5.part")
    gdown.download(
        id="1-1t1sbQXXyB-qZr-czqv2UJ4E5BQ7McI",
        output=str(temporary),
        quiet=False,
        use_cookies=False,
    )
    with h5py.File(temporary, "r") as data:
        if not {"items", "users"}.issubset(data.keys()):
            raise ValueError("В файле нет items и users")
    temporary.replace(path)


def read_embeddings(embeddings_path: str, num_embeddings: int | None):
    with h5py.File(embeddings_path, "r") as data:
        items = data["items"][:num_embeddings].astype("float32")
        users = data["users"][:].astype("float32")
    if items.ndim != 2 or users.ndim != 2 or items.shape[1] != users.shape[1]:
        raise ValueError("items и users должны быть матрицами одной размерности")
    if not np.isfinite(items).all() or not np.isfinite(users).all():
        raise ValueError("В эмбеддингах есть NaN или inf")
    return items, users


def build_faiss_index(embeddings: np.ndarray, index_path: str | None = None):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    normalized = np.array(embeddings, dtype="float32", order="C", copy=True)
    faiss.normalize_L2(normalized)
    index.add(normalized)
    if index_path is not None:
        faiss.write_index(index, index_path)
    return index


def setup_qdrant(
    collection_name: str,
    port: int,
    emb_dim: int,
    recreate: bool = False,
    m: int = 50,
    ef_construct: int = 1000,
):
    client = QdrantClient("127.0.0.1", port=port, timeout=60.0)
    try:
        if client.collection_exists(collection_name):
            if not recreate:
                raise RuntimeError(
                    f"Коллекция {collection_name} уже существует. "
                    "Для её пересоздания укажите recreate=true или другое collection_name."
                )
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=emb_dim, distance=models.Distance.COSINE, on_disk=True
            ),
            hnsw_config=models.HnswConfigDiff(
                m=m, ef_construct=ef_construct, full_scan_threshold=10
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
    except Exception:
        client.close()
        raise
    return client


def upload_to_qdrant(
    client: QdrantClient, collection_name: str, embeddings: np.ndarray, batch_size: int
):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        batch = np.array(embeddings[start:stop], dtype="float32", order="C", copy=True)
        # Qdrant leaves vectors with squared norm < float32 epsilon unnormalized.
        faiss.normalize_L2(batch)
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=list(range(start, stop)), vectors=batch.tolist()
            ),
            wait=True,
        )
    actual = client.count(collection_name, exact=True).count
    if actual != len(embeddings):
        raise RuntimeError(f"В Qdrant {actual} точек, ожидалось {len(embeddings)}")


def wait_for_index(client: QdrantClient, collection_name: str, timeout: float):
    client.update_collection(
        collection_name,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )
    deadline = time.monotonic() + timeout
    while True:
        info = client.get_collection(collection_name)
        if info.optimizer_status != models.OptimizersStatusOneOf.OK:
            raise RuntimeError(f"Ошибка индексирования: {info.optimizer_status}")
        if info.status == models.CollectionStatus.RED:
            raise RuntimeError("Коллекция Qdrant в состоянии red")
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Qdrant не завершил построение HNSW за отведённое время")
        time.sleep(1)


def search_faiss(index, query_embedding: np.ndarray, k: int = 10):
    query = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query)
    scores, ids = index.search(query, k)
    return scores[0], ids[0]


def search_qdrant(
    client: QdrantClient,
    query_embedding: np.ndarray,
    collection_name: str,
    k: int = 10,
    hnsw_ef: int = 1600,
    exact: bool = False,
):
    query = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query)
    result = client.query_points(
        collection_name=collection_name,
        query=query[0].tolist(),
        limit=k,
        search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=exact),
        with_payload=False,
        with_vectors=False,
    )
    return [(hit.score, hit.id) for hit in result.points]


def recall_at_k(found: np.ndarray, truth: np.ndarray):
    if found.shape != truth.shape or truth.ndim != 2 or not truth.size:
        raise ValueError("Результат и эталон должны иметь одинаковую непустую форму (queries, k)")
    if (truth < 0).any():
        raise ValueError("В эталоне есть отсутствующие соседи")
    return float(np.mean([
        len(set(predicted) & set(expected)) / len(expected)
        for predicted, expected in zip(found, truth)
    ]))


def benchmark(search, queries, truth, repeats, warmup):
    for query in queries[:warmup]:
        search(query)
    elapsed = []
    recalls = []
    rng = np.random.default_rng(42)
    for _ in range(repeats):
        order = rng.permutation(len(queries))
        found = []
        for row in order:
            start = time.perf_counter()
            ids = search(queries[row])
            elapsed.append((time.perf_counter() - start) * 1000)
            found.append(ids)
        recalls.append(recall_at_k(np.asarray(found), truth[order]))
    return {
        "recall_at_k": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
    }


def load_config(overrides: list[str]):
    config = OmegaConf.load(Path(__file__).with_name("conf.yaml"))
    OmegaConf.set_struct(config, True)
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


def main(conf: DictConfig):
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if conf.num_queries < 1 or conf.repeats < 1 or conf.num_neighbours < 1:
        raise ValueError("num_queries, repeats и num_neighbours должны быть положительными")
    if conf.hnsw_ef < conf.num_neighbours:
        raise ValueError("hnsw_ef должен быть не меньше num_neighbours")
    directory = Path(__file__).resolve().parent
    data_path = directory / conf.embeddings_path
    download_data(data_path)
    items, users = read_embeddings(data_path, conf.num_embeddings)
    if len(items) < conf.num_neighbours or len(users) < conf.num_queries:
        raise ValueError("Недостаточно объектов или пользователей для выбранных параметров")
    rows = np.random.default_rng(conf.seed).choice(len(users), conf.num_queries, replace=False)
    queries = np.ascontiguousarray(users[rows])
    del users
    if (np.linalg.norm(queries, axis=1) == 0).any():
        raise ValueError("У выбранного пользователя нулевой вектор")
    faiss.omp_set_num_threads(conf.faiss_threads)
    index = build_faiss_index(items)
    normalized_queries = queries.copy()
    faiss.normalize_L2(normalized_queries)
    _, truth = index.search(normalized_queries, conf.num_neighbours)

    client = setup_qdrant(
        conf.collection_name, conf.port, items.shape[1], conf.recreate,
        conf.hnsw_m, conf.ef_construct,
    )
    try:
        print(f"Загрузка {len(items)} объектов в Qdrant…", flush=True)
        upload_to_qdrant(client, conf.collection_name, items, conf.upload_batch_size)
        print("Построение HNSW…", flush=True)
        wait_for_index(client, conf.collection_name, conf.index_timeout)
        searches = {
            "FAISS FlatIP": lambda q: search_faiss(index, q, conf.num_neighbours)[1],
            "Qdrant exact": lambda q: [hit[1] for hit in search_qdrant(
                client, q, conf.collection_name, conf.num_neighbours, exact=True
            )],
            "Qdrant HNSW": lambda q: [hit[1] for hit in search_qdrant(
                client, q, conf.collection_name, conf.num_neighbours, hnsw_ef=conf.hnsw_ef
            )],
        }
        results = {}
        for name, search in searches.items():
            print(f"Замер {name}…", flush=True)
            results[name] = benchmark(search, queries, truth, conf.repeats, conf.warmup)
    finally:
        client.close()

    print(f"\n| Метод | Recall@{conf.num_neighbours} | p50, мс | p95, мс |")
    print("|---|---:|---:|---:|")
    best = {key: (max if key == "recall_at_k" else min)(r[key] for r in results.values())
            for key in next(iter(results.values()))}
    for name, result in results.items():
        values = []
        for key, value in result.items():
            formatted = f"{value:.2%}" if key == "recall_at_k" else f"{value:.2f}"
            values.append(f"**{formatted}**" if value == best[key] else formatted)
        print(f"| {name} | {' | '.join(values)} |")
    output = directory / "results" / "benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({
        "config": OmegaConf.to_container(conf, resolve=True),
        "num_items": len(items), "query_user_rows": rows.tolist(), "results": results,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск рекомендаций: FAISS и Qdrant")
    parser.add_argument(
        "overrides", nargs="*", metavar="key=value",
        help="Параметры conf.yaml, например recreate=true",
    )
    main(load_config(parser.parse_args().overrides))
