import numpy as np
import uvicorn
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import traceback
import zstd


def compress_data(data: bytes) -> bytes:
    # Miner to improve this function
    try:
        if not data:
            raise ValueError("Empty data received")
        compressed = zstd.compress(data, 3)
        return compressed
    except Exception as e:
        raise ValueError(f"Compression failed: {str(e)}") from e


def decompress_data(data: bytes) -> bytes:
    # Miner to improve this function
    try:
        if not data:
            raise ValueError("Empty data received")
        decompressed = zstd.decompress(data)
        return decompressed
    except Exception as e:
        raise ValueError(f"Decompression failed: {str(e)}") from e


def _validate(data: bytes) -> dict:
    # Validate that data compresses and decompresses correctly
    # Returns: (is_valid, compression_efficiency, cosine_similarity)
    input_array = np.frombuffer(data, dtype=np.uint8)
    compressed = compress_data(data)
    decompressed = decompress_data(compressed)
    output_array = np.frombuffer(decompressed, dtype=np.uint8)

    is_valid = np.array_equal(input_array, output_array)
    compression_efficiency = 1 - (len(compressed) / len(data))
    a = input_array.astype(np.float64)
    b = output_array.astype(np.float64)
    if np.linalg.norm(a) == 0:
        if np.linalg.norm(b) == 0:
            cosine_similarity = 1.0
        else:
            cosine_similarity = 0.0
    else:
        cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "is_valid": is_valid,
        "compression_efficiency": compression_efficiency,
        "cosine_similarity": float(cosine_similarity),
    }


def make_app() -> FastAPI:
    app = FastAPI(title="Matrix Compression Miner API")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/compress")
    async def compress(file: UploadFile = File(...)):
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="No data received in file")
            # Log data info for debugging (first 50 bytes)
            print(f"DEBUG: Received {len(data)} bytes, first 50 bytes: {data[:50].hex()}")
            compressed = compress_data(data)
            return Response(content=compressed, media_type="application/octet-stream")
        except HTTPException:
            raise
        except Exception as e:
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR in /compress: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

    @app.post("/decompress")
    async def decompress(file: UploadFile = File(...)):
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="No data received in file")
            # Log data info for debugging (first 50 bytes)
            print(f"DEBUG: Received {len(data)} bytes, first 50 bytes: {data[:50].hex()}")
            decompressed = decompress_data(data)
            return Response(content=decompressed, media_type="application/octet-stream")
        except HTTPException:
            raise
        except Exception as e:
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR in /decompress: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")
