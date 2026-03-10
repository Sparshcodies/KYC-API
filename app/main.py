import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import init_db, close_db
# from app.routes.kyc_validation import router as kyc_router
from app.routes.embeddings_store import router as embedding_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)

# app.include_router(kyc_router, prefix="/kyc")
app.include_router(embedding_router, prefix="/embedding")