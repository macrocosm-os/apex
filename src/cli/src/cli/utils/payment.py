"""Helpers for the on-chain payment flow used by `apex submit`."""

from typing import Optional


def canonicalize_block_hash(substrate, receipt_block_hash: str) -> tuple[str, Optional[str]]:
    """
    Resolve the canonical (finalized) block hash for a payment receipt.

    Returns `(block_hash, warning_message)`:
      - block_hash: the canonical hash if it could be resolved and differs
        from the receipt; otherwise the receipt hash.
      - warning_message: human-readable note when the receipt hash was
        replaced (or when re-resolution failed unexpectedly), else None.

    The function is a no-op for the happy path (receipt hash already canonical).
    Failures to look up the block are swallowed — we fall through to the
    receipt hash and let the orchestrator surface any downstream error.
    """
    try:
        block = substrate.get_block(block_hash=receipt_block_hash)
    except Exception:
        return receipt_block_hash, None
    if not block:
        return receipt_block_hash, None

    block_number = block.get("header", {}).get("number")
    if block_number is None:
        return receipt_block_hash, None

    try:
        canonical_hash = substrate.get_block_hash(block_number)
    except Exception as e:
        return (
            receipt_block_hash,
            f"Could not re-resolve canonical block hash for block {block_number}: {e}. "
            f"Proceeding with receipt hash.",
        )

    if not canonical_hash or canonical_hash == receipt_block_hash:
        return receipt_block_hash, None

    return (
        canonical_hash,
        f"Receipt hash {receipt_block_hash} differs from canonical hash at "
        f"block {block_number} ({canonical_hash}); using canonical.",
    )
