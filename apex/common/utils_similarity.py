import re

from loguru import logger
from rouge import Rouge


def compute_rouge_score(reference: str, generation: str) -> float:
    """Calculate the ROUGE score between a reference and a generation."""
    if not reference or not generation:
        return 0.0

    if not reference.strip() or not generation.strip():
        return 0.0

    try:
        rouge = Rouge()
        scores = rouge.get_scores(generation, reference)
        rouge_l_f1 = scores[0]["rouge-l"]["f"]
        return float(rouge_l_f1)

    except Exception as e:
        logger.error(f"Error calculating ROUGE score: {e}")
        return 0.0


async def compute_similarity_score(reference: str, generation: str) -> float:
    """Calculate the similarity score between a reference and a generation."""
    # TODO: Implement API call for similarity scoring
    return 0.0


def compute_character_count_diff(reference: str, generation: str) -> float:
    """Calculate the percentage difference in character count between a reference and a generation.

    Returns the percentage increase from the shorter string to the longer string.
    """
    if not reference or not generation:
        return 0.0

    ref_count = len(reference)
    gen_count = len(generation)

    if ref_count == 0 and gen_count == 0:
        return 0.0

    shorter = min(ref_count, gen_count)
    longer = max(ref_count, gen_count)

    if shorter == 0:
        return 0.0

    difference = (longer - shorter) / shorter
    return 1 - difference


def compute_sentence_length_diff(reference: str, generation: str) -> float:
    """Calculate the percentage difference in sentence count between a reference and a generation.

    Returns the percentage increase from the shorter string to the longer string.
    """
    if not reference or not generation:
        return 0.0

    # Count sentences by splitting on sentence-ending punctuation.
    ref_sentences = len([s.strip() for s in re.split(r"[.!?]+", reference) if s.strip()])
    gen_sentences = len([s.strip() for s in re.split(r"[.!?]+", generation) if s.strip()])

    if ref_sentences == 0 and gen_sentences == 0:
        return 0.0

    shorter = min(ref_sentences, gen_sentences)
    longer = max(ref_sentences, gen_sentences)

    if shorter == 0:
        return 0.0

    difference = (longer - shorter) / shorter
    return 1 - difference
