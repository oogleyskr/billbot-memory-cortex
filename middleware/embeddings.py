"""Embeddings utilities: compute, serialize, and compare vectors via nomic-embed."""

import logging
import math
import struct
from typing import Optional

import httpx

logger = logging.getLogger("memory-cortex.embeddings")

# Default endpoint for the nomic-embed service (RTX 3090, port 8105)
DEFAULT_EMBED_URL = "http://localhost:8105/embed"
DEFAULT_TIMEOUT = 30.0


async def compute_embedding(
    text: str,
    embed_url: str = DEFAULT_EMBED_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[list[float]]:
    """Compute an embedding vector for a single text string.

    POSTs to the nomic-embed service and returns the embedding as a list of
    floats. Returns None on any failure (network, timeout, bad response).
    """
    if not text or not text.strip():
        logger.warning("compute_embedding called with empty text")
        return None

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                embed_url,
                json={"input": text},
            )
            response.raise_for_status()
            data = response.json()

            # The service returns OpenAI-compatible format:
            # {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [...]}]}
            items = data.get("data")
            if not items or not isinstance(items, list):
                logger.error("No 'data' array in response: %s", list(data.keys()))
                return None

            embedding = items[0].get("embedding")
            if embedding is None:
                logger.error("No 'embedding' key in data[0]: %s", list(items[0].keys()))
                return None

            if not isinstance(embedding, list) or len(embedding) == 0:
                logger.error("Invalid embedding format: type=%s len=%s",
                             type(embedding).__name__,
                             len(embedding) if isinstance(embedding, list) else "N/A")
                return None

            return embedding

        except httpx.TimeoutException:
            logger.error("Timeout computing embedding (%.1fs)", timeout)
            return None
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %d from embed service: %s", e.response.status_code, e)
            return None
        except httpx.HTTPError as e:
            logger.error("HTTP error computing embedding: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error computing embedding: %s", e)
            return None


async def compute_embeddings_batch(
    texts: list[str],
    embed_url: str = DEFAULT_EMBED_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Optional[list[float]]]:
    """Compute embeddings for a batch of texts.

    Calls compute_embedding for each text individually. Returns a list of the
    same length as the input, where each element is either a list[float] or
    None if that particular embedding failed.
    """
    results: list[Optional[list[float]]] = []
    for text in texts:
        embedding = await compute_embedding(text, embed_url, timeout)
        results.append(embedding)
    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Returns 0.0 if either vector has zero
    magnitude (to avoid division by zero).
    """
    if len(a) != len(b):
        logger.warning("Vector length mismatch: %d vs %d", len(a), len(b))
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a float vector to bytes for SQLite BLOB storage.

    Uses struct.pack with float32 ('f') format. The first 4 bytes store the
    dimension count as an unsigned int, followed by N float32 values.
    """
    n = len(embedding)
    # Pack: [uint32 dimension count] [float32 * N]
    return struct.pack(f"<I{n}f", n, *embedding)


def deserialize_embedding(data: bytes) -> Optional[list[float]]:
    """Deserialize bytes back to a list of floats.

    Returns None if the data is invalid or corrupted.
    """
    if data is None or len(data) < 4:
        logger.warning("Cannot deserialize embedding: data is None or too short")
        return None

    try:
        # Read dimension count
        (n,) = struct.unpack("<I", data[:4])
        expected_size = 4 + n * 4  # 4 bytes header + N * 4 bytes per float32
        if len(data) != expected_size:
            logger.error(
                "Embedding data size mismatch: expected %d bytes (n=%d), got %d",
                expected_size, n, len(data),
            )
            return None

        values = struct.unpack(f"<{n}f", data[4:])
        return list(values)

    except struct.error as e:
        logger.error("Failed to deserialize embedding: %s", e)
        return None
