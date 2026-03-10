from fastapi import APIRouter, HTTPException
from app.database import insert_embeddings, fetch_embeddings
from app.schema.kyc_schema import StoreEmbeddingsRequest, StoreEmbeddingsResponse, RetrieveEmbeddingsRequest, RetrieveEmbeddingsResponse

router = APIRouter()

@router.post("/store_embeddings", response_model=StoreEmbeddingsResponse)
async def store_embeddings(payload: StoreEmbeddingsRequest):
    if not payload.embeddings:
        raise HTTPException(status_code=400, detail="Embeddings list is empty")
    saved = await insert_embeddings(payload.user_id, payload.embeddings)
    return {
        "status": "stored",
        "embeddings_saved": saved
    }

@router.post("/retrieve_embeddings", response_model=RetrieveEmbeddingsResponse)
async def retrieve_embeddings(payload: RetrieveEmbeddingsRequest):
    embeddings = await fetch_embeddings(payload.user_id)
    if not embeddings:
        raise HTTPException(
            status_code=404,
            detail="User embeddings not found"
        )
    return {
        "user_id": payload.user_id,
        "embeddings": embeddings
    }