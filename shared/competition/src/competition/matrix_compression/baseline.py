import numpy as np
import lzma
import json
import struct
import base64
from pydantic import BaseModel
import io


_MAGIC = b"AMZ\x00"


class CompressionInputDataSchema(BaseModel):
    data_to_compress_base64: str
    expected_output_filepath: str


class DecompressionInputDataSchema(BaseModel):
    data_to_decompress_base64: str
    expected_output_filepath: str


def compress(input_data: CompressionInputDataSchema) -> None:
    """
    Compress an activation array to disk with sparsity-aware quantization.
    """
    mode: str = "lossless"  # "lossless", "q8", "q4"
    epsilon: float = 0.0  # treat |x|<=epsilon as 0 for sparsity mask (use 1e-8..1e-6 for float noise)
    lzma_preset: int = 9
    lzma_extreme: bool = True

    # Decode the base64 data back to numpy array inline
    arr_bytes = base64.b64decode(input_data.data_to_compress_base64)
    buffer = io.BytesIO(arr_bytes)
    arr = np.load(buffer, allow_pickle=False)

    # Normalize storage order and endianness for portability
    arr_c = np.ascontiguousarray(arr)
    if arr_c.dtype.byteorder == ">" or (arr_c.dtype.byteorder == "=" and not np.little_endian):
        arr_c = arr_c.byteswap().newbyteorder("<")

    header = {
        "version": 1,
        "shape": tuple(arr_c.shape),
        "dtype": str(arr_c.dtype),
        "mode": mode,
        "epsilon": float(epsilon),
        "has_mask": False,
        "mask_bytes": 0,
        "nz": 0,
        "scale": None,
    }

    def _finalize_and_write(payload_bytes: bytes, extra_blobs: list[bytes]) -> None:
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        preset_val = lzma_preset | (lzma.PRESET_EXTREME if lzma_extreme else 0)
        compressed = lzma.compress(b"".join(extra_blobs + [payload_bytes]), preset=preset_val, format=lzma.FORMAT_XZ)
        with open(input_data.expected_output_filepath, "wb") as f:
            f.write(_MAGIC)
            f.write(header_len)
            f.write(header_bytes)
            f.write(compressed)

    # Validate special values for lossy modes
    has_nan_inf = np.isnan(arr_c).any() or np.isinf(arr_c).any()

    if mode == "lossless" or has_nan_inf or not np.issubdtype(arr_c.dtype, np.floating):
        # Exact: optionally bit-pack bools; otherwise raw bytes
        if arr_c.dtype == np.bool_:
            flat = arr_c.reshape(-1)
            packed = _pack_bits(flat)
            header["mode"] = "lossless-bool-packed"
            header["has_mask"] = True
            header["mask_bytes"] = len(packed)
            _finalize_and_write(b"", [packed])  # payload is just the packed bits
        else:
            payload = arr_c.tobytes(order="C")
            _finalize_and_write(payload, [])
        return

    # From here: float activations, quantization modes
    x = arr_c.astype(np.float32, copy=False)  # work in f32 for stable scaling
    # Sparsity mask (True = kept value)
    if epsilon > 0.0:
        mask = np.abs(x) > epsilon
    else:
        # exact zero is fine; many ReLU activations are exactly 0
        mask = x != 0.0

    n_bits = x.size
    nz = int(mask.sum())
    header["has_mask"] = True
    header["mask_bytes"] = int(np.ceil(n_bits / 8))
    header["nz"] = nz

    # Edge case: all zeros (or all pruned)
    if nz == 0:
        # Just write the mask; values are all zeros
        mask_blob = _pack_bits(mask)
        _finalize_and_write(b"", [mask_blob])
        return

    # Gather nonzero values
    vals = x[mask]

    # Compute symmetric scale (zero-point = 0); robust to outliers via max-abs
    amax = float(np.max(np.abs(vals)))
    if amax == 0.0:
        mask_blob = _pack_bits(mask)
        _finalize_and_write(b"", [mask_blob])
        return

    if mode == "q8":
        # Map to -127..127 (reserve -128 for saturation)
        scale = amax / 127.0
        q = np.clip(np.rint(vals / scale), -127, 127).astype(np.int8)
        header["scale"] = float(scale)
        # blobs: [mask_blob] + q.tobytes()
        mask_blob = _pack_bits(mask)
        payload = q.tobytes()
        _finalize_and_write(payload, [mask_blob])
        return

    if mode == "q4":
        # Signed 4-bit range -8..7 (common symmetric int4)
        # Using 7 in denominator yields slightly lower error for positives,
        # but we'll allow -8 saturation; pick 7 for scale (tighter fit).
        scale = amax / 7.0
        q = np.clip(np.rint(vals / scale), -8, 7).astype(np.int8)
        header["scale"] = float(scale)
        mask_blob = _pack_bits(mask)
        packed_q = _pack_int4(q)
        _finalize_and_write(packed_q, [mask_blob])
        return

    raise ValueError("mode must be 'lossless', 'q8', or 'q4'")


def decompress(input_data: DecompressionInputDataSchema) -> None:
    """
    Decompress data from base64 encoded compressed blob.
    """
    # Decode the base64 compressed data
    blob = base64.b64decode(input_data.data_to_decompress_base64)

    # Parse the blob directly
    if len(blob) < 4:
        raise ValueError("Invalid compressed data.")

    magic = blob[:4]
    if magic != _MAGIC:
        raise ValueError("Not an AMZ file or wrong version.")

    header_len = struct.unpack("<I", blob[4:8])[0]
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    compressed_data = blob[8 + header_len :]

    shape = tuple(header["shape"])
    dtype = np.dtype(header["dtype"])
    mode = header["mode"]
    has_mask = header.get("has_mask", False)

    # Lossless-bool-packed
    if mode == "lossless-bool-packed":
        n_bits = int(np.prod(shape))
        mask = _unpack_bits(compressed_data, n_bits).reshape(shape)
        result = mask.astype(np.bool_)

        # Write the decompressed result to the output file path
        buffer = io.BytesIO()
        np.save(buffer, result, allow_pickle=False)
        buffer.seek(0)
        result_bytes = buffer.getvalue()

        with open(input_data.expected_output_filepath, "wb") as f:
            f.write(result_bytes)
        return

    # Pure lossless (raw)
    if mode == "lossless":
        arr = np.frombuffer(lzma.decompress(compressed_data, format=lzma.FORMAT_XZ), dtype=dtype)
        result = arr.reshape(shape, order="C")

        # Write the decompressed result to the output file path
        buffer = io.BytesIO()
        np.save(buffer, result, allow_pickle=False)
        buffer.seek(0)
        result_bytes = buffer.getvalue()

        with open(input_data.expected_output_filepath, "wb") as f:
            f.write(result_bytes)
        return

    # Lossy modes carry: mask || payload (in that order)
    # Split out mask first
    if has_mask:
        mask_nbytes = header["mask_bytes"]
        mask_bytes = compressed_data[:mask_nbytes]
        payload = compressed_data[mask_nbytes:]
        n_bits = int(np.prod(shape))
        mask = _unpack_bits(mask_bytes, n_bits)
    else:
        payload = compressed_data
        mask = np.ones(int(np.prod(shape)), dtype=bool)

    # Decompress combined content
    decompressed = lzma.decompress(payload, format=lzma.FORMAT_XZ) if mode in ("lossless",) else payload

    # For q8/q4, payload was *not* xz-compressed separately (already compressed as a whole),
    # so we must parse based on header["nz"] only.
    nz = int(header.get("nz", 0))
    out = np.zeros(int(np.prod(shape)), dtype=np.float32)

    if nz == 0:
        result = out.reshape(shape).astype(dtype, copy=False)

        # Write the decompressed result to the output file path
        buffer = io.BytesIO()
        np.save(buffer, result, allow_pickle=False)
        buffer.seek(0)
        result_bytes = buffer.getvalue()

        with open(input_data.expected_output_filepath, "wb") as f:
            f.write(result_bytes)
        return

    if mode == "q8":
        q = np.frombuffer(decompressed, dtype=np.int8, count=nz)
        scale = float(header["scale"])
        vals = q.astype(np.float32) * scale
    elif mode == "q4":
        q = _unpack_int4(decompressed, nz).astype(np.int8)
        scale = float(header["scale"])
        vals = q.astype(np.float32) * scale
    else:
        raise ValueError(f"Unknown lossy mode: {mode}")

    out[mask] = vals
    result = out.reshape(shape).astype(dtype if np.issubdtype(dtype, np.floating) else np.float32, copy=False)

    # Write the decompressed result to the output file path
    buffer = io.BytesIO()
    np.save(buffer, result, allow_pickle=False)
    buffer.seek(0)
    result_bytes = buffer.getvalue()

    with open(input_data.expected_output_filepath, "wb") as f:
        f.write(result_bytes)


def _pack_bits(bits: np.ndarray) -> bytes:
    """
    Pack a boolean array into bytes.
    Each byte contains 8 bits, with the first bit being the least significant bit.
    """
    if len(bits) == 0:
        return b""

    # Pad to multiple of 8
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    padded_bits = np.zeros(n_bytes * 8, dtype=bool)
    padded_bits[:n_bits] = bits

    # Reshape to groups of 8 and pack
    packed = padded_bits.reshape(-1, 8)
    result = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(8):
        result |= packed[:, i].astype(np.uint8) << i

    return result.tobytes()


def _unpack_bits(data: bytes, n_bits: int) -> np.ndarray:
    """
    Unpack bytes back to a boolean array.
    """
    if n_bits == 0:
        return np.array([], dtype=bool)

    n_bytes = len(data)
    if n_bytes == 0:
        return np.zeros(n_bits, dtype=bool)

    # Convert bytes to uint8 array
    bytes_array = np.frombuffer(data, dtype=np.uint8)

    # Unpack each byte into 8 bits
    bits = np.zeros(n_bytes * 8, dtype=bool)
    for i in range(8):
        bits[i::8] = (bytes_array & (1 << i)) != 0

    # Return only the requested number of bits
    return bits[:n_bits]


def _pack_int4(values: np.ndarray) -> bytes:
    """
    Pack int8 values into 4-bit packed format.
    Each byte contains two 4-bit values (nibbles).
    """
    if len(values) == 0:
        return b""

    # Ensure values are in valid 4-bit range (-8 to 7)
    values = np.clip(values, -8, 7)

    # Convert to unsigned 4-bit representation
    # -8 becomes 0, -7 becomes 1, ..., 7 becomes 15
    unsigned_vals = (values + 8) & 0xF

    # Pad to even length
    if len(unsigned_vals) % 2 == 1:
        unsigned_vals = np.append(unsigned_vals, 0)

    # Pack pairs of 4-bit values into bytes
    packed = np.zeros(len(unsigned_vals) // 2, dtype=np.uint8)
    packed |= unsigned_vals[::2] & 0xF  # Low nibble
    packed |= (unsigned_vals[1::2] & 0xF) << 4  # High nibble

    return packed.tobytes()


def _unpack_int4(data: bytes, n_values: int) -> np.ndarray:
    """
    Unpack 4-bit packed data back to int8 values.
    """
    if n_values == 0:
        return np.array([], dtype=np.int8)

    if len(data) == 0:
        return np.zeros(n_values, dtype=np.int8)

    # Convert bytes to uint8 array
    bytes_array = np.frombuffer(data, dtype=np.uint8)

    # Unpack each byte into two 4-bit values
    n_bytes = len(bytes_array)
    values = np.zeros(n_bytes * 2, dtype=np.int8)

    # Extract low nibbles (bits 0-3)
    values[::2] = (bytes_array & 0xF) - 8
    # Extract high nibbles (bits 4-7)
    values[1::2] = ((bytes_array >> 4) & 0xF) - 8

    # Return only the requested number of values
    return values[:n_values]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--input-file", type=str)
    args = parser.parse_args()

    if args.compress:
        with open(args.input_file, "r") as f:
            input_data = CompressionInputDataSchema.model_validate_json(f.read())

        compress(input_data=input_data)
    elif args.decompress:
        with open(args.input_file, "r") as f:
            input_data = DecompressionInputDataSchema.model_validate_json(f.read())
        decompress(input_data=input_data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
