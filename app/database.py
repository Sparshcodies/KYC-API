import numpy as np
import psycopg
from psycopg_pool import AsyncConnectionPool
from app.config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER


DB_DSN = (
    f"host={DB_HOST} "
    f"dbname={DB_NAME} "
    f"user={DB_USER} "
    f"password={DB_PASS} "
    f"port={DB_PORT}"
)

pool = AsyncConnectionPool(
    conninfo=DB_DSN,
    min_size=1,
    max_size=5,
    open=False,
)


async def init_db():
    await pool.open()
    print("DB pool started")


async def close_db():
    await pool.close()
    print("DB pool closed")


async def insert_embeddings(user_id: str, embeddings):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for emb in embeddings:
                await cur.execute(
                    """
                    INSERT INTO kyc_embeddings (user_id, embedding)
                    VALUES (%s, %s)
                    """,
                    (user_id, emb.tolist()),
                )
        await conn.commit()


async def fetch_embeddings(user_id: str):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT embedding FROM kyc_embeddings WHERE user_id=%s",
                (user_id,),
            )
            rows = await cur.fetchall()

    return [np.array(r[0], dtype=np.float32) for r in rows]

async def fetch_embeddings_for_users(user_ids):
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT user_id, embedding
                FROM kyc_embeddings
                WHERE user_id = ANY(%s)
                """,
                (user_ids,)
            )
            rows = await cur.fetchall()

    user_gallery = {}

    for user_id, emb in rows:
        user_gallery.setdefault(user_id, []).append(
            np.array(emb, dtype=np.float32)
        )

    return user_gallery