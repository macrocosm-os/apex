"""Baseline miner solution for text clustering competition.

Uses TF-IDF vectorization + KMeans clustering. Fast but less accurate
than embedding-based methods. Miners should improve upon this.

Usage:
    python baseline.py --port 8001
"""

import argparse
import re

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class ClusterRequest(BaseModel):
    texts: list[str]


class ClusterResponse(BaseModel):
    cluster_ids: list[int]


def preprocess_text(text: str) -> str:
    """Basic text preprocessing for social media posts."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def estimate_num_clusters(n_samples: int) -> int:
    """Estimate cluster count using sqrt(n/2) heuristic."""
    k = int(np.sqrt(n_samples / 2))
    return max(2, min(k, 100))


def cluster_texts(texts: list[str]) -> list[int]:
    """Cluster texts using TF-IDF + KMeans."""
    if len(texts) < 2:
        return [0] * len(texts)

    processed = [preprocess_text(t) for t in texts]
    empty_mask = [len(t.strip()) == 0 for t in processed]
    non_empty_texts = [t for t, is_empty in zip(processed, empty_mask) if not is_empty]

    if len(non_empty_texts) < 2:
        return [0] * len(texts)

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        dtype=np.float32,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(non_empty_texts)
    except ValueError:
        return [0] * len(texts)

    n_clusters = estimate_num_clusters(len(non_empty_texts))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=3,
    )
    non_empty_labels = kmeans.fit_predict(tfidf_matrix)

    labels = []
    non_empty_idx = 0
    for is_empty in empty_mask:
        if is_empty:
            labels.append(-1)
        else:
            labels.append(int(non_empty_labels[non_empty_idx]))
            non_empty_idx += 1

    return labels


def make_app() -> FastAPI:
    app = FastAPI(title="Text Clustering Miner")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster(request: ClusterRequest) -> ClusterResponse:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        cluster_ids = cluster_texts(request.texts)
        return ClusterResponse(cluster_ids=cluster_ids)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")
