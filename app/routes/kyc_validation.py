from fastapi import APIRouter, HTTPException
from app.utils.storage import fetch_video
from app.database import insert_embeddings, fetch_embeddings, fetch_all_embeddings
from app.utils.face_verifier import FaceVerifier
from app.schema.kyc_schema import RegisterRequest, RegisterResponse, VerifyRequest, VerifyResponse, IdentifyRequest, IdentifyResponse

router = APIRouter()

verifier = FaceVerifier()

@router.post("/register_reference", response_model=RegisterResponse)
async def register_reference(payload: RegisterRequest):
    video_path = fetch_video(payload.video_url)
    embeddings = verifier.build_gallery_from_video(video_path)
    if not embeddings:
        raise HTTPException(status_code=400, detail="No faces detected")
    await insert_embeddings(payload.user_id, embeddings)
    return {
        "status": "stored",
        "embeddings_saved": len(embeddings)
    }

@router.post("/verify", response_model=VerifyResponse)
async def verify(payload: VerifyRequest):
    video_path = fetch_video(payload.video_url)
    reference_embeddings = await fetch_embeddings(payload.user_id)
    if not reference_embeddings:
        raise HTTPException(status_code=404, detail="User embeddings not found")
    result = verifier.compare_gallery(reference_embeddings, video_path)
    return {
        "user_id": payload.user_id,
        "result": result["result"],
        "similarity_score": result["probability"]
    }
    
@router.post("/identify_video", response_model=IdentifyResponse)
async def identify_video(payload: IdentifyRequest):
    video_path = fetch_video(payload.video_url)
    all_embeddings = await fetch_all_embeddings()
    if not all_embeddings:
        raise HTTPException(
            status_code=404,
            detail="No registered users found"
        )
    output_path = verifier.identify_faces_and_annotate(
        video_path,
        all_embeddings,
        payload.output_path
    )
    return {"output_video": output_path}