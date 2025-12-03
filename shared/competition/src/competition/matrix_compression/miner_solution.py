# Baseline submission just saves the array as-is
import numpy as np
import base64
from pydantic import BaseModel
import io


class CompressionInputDataSchema(BaseModel):
    data_to_compress_base64: str
    expected_output_filepath: str


class DecompressionInputDataSchema(BaseModel):
    data_to_decompress_base64: str
    expected_output_filepath: str


class MatrixCompressTaskSchema(BaseModel):
    task_name: str
    input: CompressionInputDataSchema | DecompressionInputDataSchema


class MatrixCompressEvalInputDataSchema(BaseModel):
    tasks: list[MatrixCompressTaskSchema]


def compress(input_data: CompressionInputDataSchema) -> None:
    """
    Compress an activation array to disk with sparsity-aware quantization.
    Implement your compression algorithm here.
    """
    # TODO: Implement your compression algorithm
    # This is a placeholder that just saves the array as-is

    # Decode the base64 data back to numpy array inline
    arr_bytes = base64.b64decode(input_data.data_to_compress_base64)
    buffer = io.BytesIO(arr_bytes)
    arr = np.load(buffer, allow_pickle=False)

    # Placeholder: just save the array as a simple numpy file
    np.save(input_data.expected_output_filepath, arr)


def decompress(input_data: DecompressionInputDataSchema) -> None:
    """
    Decompress data from base64 encoded compressed blob.
    Implement your decompression algorithm here.
    """
    # TODO: Implement your decompression algorithm
    # This is a placeholder that just loads the array as-is

    # Decode the base64 compressed data
    blob = base64.b64decode(input_data.data_to_decompress_base64)

    # Placeholder: assume the blob is a numpy array saved directly
    buffer = io.BytesIO(blob)
    arr = np.load(buffer, allow_pickle=False)

    # Write the decompressed result to the output file path
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    buffer.seek(0)
    result_bytes = buffer.getvalue()

    with open(input_data.expected_output_filepath, "wb") as f:
        f.write(result_bytes)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--decompress", action="store_true")
    parser.add_argument("--input-file", type=str)
    args = parser.parse_args()

    if args.compress:
        with open(args.input_file, "r") as f:
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(f.read())

        for task in input_data.tasks:
            compress(input_data=task.input)
    elif args.decompress:
        with open(args.input_file, "r") as f:
            input_data = MatrixCompressEvalInputDataSchema.model_validate_json(f.read())

        for task in input_data.tasks:
            decompress(input_data=task.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
