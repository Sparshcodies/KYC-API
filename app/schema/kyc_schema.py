from pydantic import BaseModel
from typing import Optional, List

class RegisterRequest(BaseModel):
    user_id: str
    video_url: str

class RegisterResponse(BaseModel):
    status: str
    embeddings_saved: int
    
class VerifyRequest(BaseModel):
    user_id: str
    video_url: str

class VerifyResponse(BaseModel):
    user_id: str
    result: str
    similarity_score: float | None = None
    
class IdentifyUsersRequest(BaseModel):
    user_ids: List[str]
    video_url: str
    output_path: str

class IdentifyUsersResponse(BaseModel):
    output_video: str
    
class StoreEmbeddingsRequest(BaseModel):
    user_id: str
    embeddings: List[List[float]]

class StoreEmbeddingsResponse(BaseModel):
    status: str
    embeddings_saved: int

class RetrieveEmbeddingsRequest(BaseModel):
    user_id: str
    
class RetrieveEmbeddingsResponse(BaseModel):
    user_id: str
    embeddings: List[List[float]]