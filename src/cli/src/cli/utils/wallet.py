from bittensor_wallet import Keypair
import json
from pathlib import Path


def load_keypair_from_file(hotkey_file_path: str) -> Keypair:
    """Load a Keypair from a bittensor key file"""
    hotkey_path = Path(hotkey_file_path)

    if not hotkey_path.exists():
        raise FileNotFoundError(f"Key file not found: {hotkey_file_path}")

    # Read the key file
    with open(hotkey_path, "r") as f:
        key_data = json.load(f)

    # Create Keypair from the private key
    private_key = key_data.get("privateKey")
    if not private_key:
        raise ValueError("No private key found in key file")

    # Convert hex string to bytes if needed
    if isinstance(private_key, str):
        # Remove 0x prefix if present
        if private_key.startswith("0x"):
            private_key = private_key[2:]
        # Convert to bytes first, then back to hex string for bittensor
        private_key_bytes = bytes.fromhex(private_key)
        private_key = private_key_bytes.hex()

    return Keypair.create_from_private_key(private_key)
