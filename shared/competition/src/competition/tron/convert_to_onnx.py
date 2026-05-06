#!/usr/bin/env python
"""
Convert a submitted TorchScript Tron model into ONNX so it can be served to
browsers. Runs inside the tron sandbox (which already has CPU torch).

Reads the .pt at /app/<model_filename>, writes the .onnx at
/app/<output_filename>, and exits non-zero on failure so the worker can detect
conversion errors.
"""

import argparse
import sys
from pathlib import Path

import onnx
import torch


def convert(input_path: str, output_path: str, grid_size: int) -> None:
    model = torch.jit.load(input_path, map_location="cpu")
    model.eval()
    dummy_input = torch.randn(1, 5, grid_size, grid_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    # Reject structurally malformed exports here rather than letting the
    # browser fail to load the model.
    onnx.checker.check_model(onnx.load(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Tron TorchScript model to ONNX.")
    parser.add_argument("--input", required=True, help="Path to the .pt TorchScript archive")
    parser.add_argument("--output", required=True, help="Path to write the .onnx file to")
    parser.add_argument("--grid-size", type=int, default=32, help="Grid edge length (default: 32)")
    args = parser.parse_args()

    try:
        convert(args.input, args.output, args.grid_size)
    except Exception as exc:
        # Remove any partial/invalid output so the worker's file-existence
        # check correctly treats this as a failure
        Path(args.output).unlink(missing_ok=True)
        print(f"ONNX conversion failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Converted {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
