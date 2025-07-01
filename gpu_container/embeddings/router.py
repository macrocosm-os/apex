from typing import List

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel

from gpu_container.decorators import require_resource

router = APIRouter()


class EmbeddingRequest(BaseModel):
    input: List[str]


class Embedding(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
@require_resource()
async def get_embeddings(request: Request, body: EmbeddingRequest):
    """Generate embeddings for a list of texts."""
    model = request.app.state.embeddings_model
    model_id = request.app.state.embeddings_model_id

    # Generate embeddings
    embeddings = model.encode(body.input, to_numpy=True)

    # Ensure embeddings are a list of lists of floats
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    response_data = [Embedding(index=i, embedding=embedding) for i, embedding in enumerate(embeddings)]

    return EmbeddingResponse(data=response_data, model=model_id)
