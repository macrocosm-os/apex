import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model files")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to save the model files",
    )

    args = parser.parse_args()

    print(f"Downloading Model {args.model_name}, files downloaded to {args.model_path}")

    snapshot_download(
        repo_id=args.model_name,
        local_dir=args.model_path
    )

    print(f"Model files downloaded to {args.model_path}")