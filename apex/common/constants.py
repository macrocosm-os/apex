MAX_TOKENS: int = 2048
TEMPERATURE: float = 0.1
WEBPAGE_MAXSIZE: int = 500
VALIDATOR_REFERENCE_LABEL = "Validator"


_ENGLISH_WORDS: tuple[str, ...] | None = None
_ENGLISH_WORDS_PATH: str = "apex/data/combined_vocab.txt"


def get_english_words() -> tuple[str, ...]:
    global _ENGLISH_WORDS

    if _ENGLISH_WORDS is None:
        with open(_ENGLISH_WORDS_PATH, encoding="utf-8") as fh:
            _ENGLISH_WORDS = tuple(word.strip() for word in fh if word.strip())

    return _ENGLISH_WORDS
