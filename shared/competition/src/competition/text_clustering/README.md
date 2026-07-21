# Text Clustering Competition

## Overview

Challenge miners to develop fast, CPU-only text clustering algorithms that approximate industry-standard NLP pipelines (sentence embeddings + UMAP + HDBScan) on real social media data from Bittensor Subnet 13 (Gravity).

## How It Works

Each round runs **3 independent subsets** (each a fresh batch of texts with its own
ground truth). For every subset:
1. Texts are collected from X and Reddit via **Gravity** (SN13 batch collection)
2. Ground truth labels are pre-computed using embedding-based clustering
3. Your solution receives raw social media texts via HTTP
4. You return cluster assignments
5. Scoring compares your clusters to ground truth using ARI + NMI

Your round score is the mean across the 3 subsets.

## Data Source

Texts are real social media posts collected by **Bittensor Subnet 13** miners through the
[Gravity](https://docs.macrocosmos.ai/developers/readme/gravity) decentralized data network.

- **Platforms**: X (Twitter) and Reddit
- **Topics vary each round**: AI, crypto, politics, sports, science, etc.
- **Volume**: 1K-50K texts per round
- **Collection**: Gravity Tasks run for up to 7 days, crawling specified topics across the SN13 miner network

## API Interface

Your solution must implement a FastAPI server with these endpoints:

### Health Check
```
GET /health
Response: {"status": "healthy"}
```

### Clustering Endpoint
```
POST /cluster
Request: {"texts": ["text1", "text2", ...]}
Response: {"cluster_ids": [0, 1, 0, 2, ...]}
```

Each `cluster_id` is an integer. Use `-1` for noise/unclassifiable texts.

## Scoring

| Metric | Range | Description |
|--------|-------|-------------|
| Adjusted Rand Index (ARI) | -1 to 1 | Similarity between clusterings, adjusted for chance |
| Normalized Mutual Information (NMI) | 0 to 1 | Mutual dependence between assignments |

**Per-subset score** = average of clamped ARI and NMI:

```
ari_normalized = max(0.0, ari)          # negative ARI (random/degenerate) → 0
combined       = (ari_normalized + nmi) / 2
```

Range: 0–1. ARI is clamped at 0 (not remapped via `(ari + 1) / 2`) so a random or
degenerate clustering scores 0 rather than ~0.25.

**Round score** = mean of the `combined` scores across the round's subsets.

## Constraints

- **CPU only** — no GPU available in the sandbox
- **Time limit** — 90 seconds for the clustering request
- **No internet** — cannot download models at runtime
- **Memory** — standard container limits (1.5 GB)
- **Max submission size** — 50,000 characters

## Exploring Data

Use the [Macrocosmos Dataverse CLI](https://github.com/macrocosm-os/dataverse-cli)
to explore the kind of social media text your algorithm will cluster:

```bash
# Install
cargo install dataverse-cli

# Get a free API key
dv auth    # key from https://app.macrocosmos.ai/account?tab=api-keys

# Search X (Twitter) posts
dv -o json search x -k AI -l 100
dv -o json search x -k bitcoin,crypto -l 200

# Search Reddit
dv -o json search reddit -k MachineLearning -l 50

# Extract just the text content
dv -o json search x -k bittensor -l 50 | jq '.[].content'
```

Or use the Python SDK for programmatic access:
```bash
pip install macrocosmos
```
```python
from macrocosmos import Sn13Client
client = Sn13Client(api_key="your-key")
response = client.on_demand_data(source="X", keywords=["AI", "technology"], limit=100)
texts = [post["content"] for post in response["data"]]
```

## Getting Started

1. Copy `baseline.py` as your starting point
2. Improve the clustering algorithm
3. Test locally:
   ```bash
   pip install -r dockerfiles/requirements.txt
   python baseline.py --port 8001

   # Test with curl
   curl -X POST http://localhost:8001/cluster \
     -H "Content-Type: application/json" \
     -d '{"texts": ["AI is transforming tech", "Machine learning rocks", "The weather is nice today", "It is sunny outside"]}'
   ```
4. Submit via `apex submit`

## Ideas for Improvement

- Lightweight embeddings (fastText, word2vec, distilled models)
- Better text preprocessing for social media (emoji handling, slang normalization)
- Adaptive cluster count estimation (silhouette analysis, gap statistic)
- Approximate nearest neighbors for density-based clustering
- Spectral clustering or hierarchical methods
- Pre-trained lightweight topic models
