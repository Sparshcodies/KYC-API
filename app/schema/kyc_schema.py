from pydantic import BaseModel
from typing import Optional

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
    
class IdentifyRequest(BaseModel):
    video_url: str
    output_path: str

class IdentifyResponse(BaseModel):
    output_video: str