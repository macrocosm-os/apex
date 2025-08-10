MAX_TOKENS: int = 2048
TEMPERATURE: float = 0.1
WEBPAGE_MAXSIZE: int = 500
VALIDATOR_REFERENCE_LABEL = "Validator"
VALIDATOR_VERIFIED_HOTKEYS = {
    "5CGLCBndTR1BvQZzn429ckT8GyxduzyjMgt4K1UVTYa8gKfb": "167.99.236.79:8001",  # Macrocosmos.
    "5CUbyC2Ez7tWYYmnFSSwjqkw26dFNo9cXH8YmcxBSfxi2XSG": None,  # Yuma.
    "5C8Em1kDZi5rxgDN4zZtfoT7dUqJ7FFbTzS3yTP5GPgVUsn1": None,  # RoundTable21.
    "5HmkM6X1D3W3CuCSPuHhrbYyZNBy2aGAiZy9NczoJmtY25H7": None,  # Crucible.
    "5GeR3cDuuFKJ7p66wKGjY65MWjWnYqffq571ZMV4gKMnJqK5": None,  # OTF.
    "5D1saVvssckE1XoPwPzdHrqYZtvBJ3vESsrPNxZ4zAxbKGs1": None,  # Rizzo.
}


_ENGLISH_WORDS: tuple[str, ...] | None = None
_ENGLISH_WORDS_PATH: str = "apex/data/combined_vocab.txt"


def get_english_words() -> tuple[str, ...]:
    global _ENGLISH_WORDS

    if _ENGLISH_WORDS is None:
        with open(_ENGLISH_WORDS_PATH, encoding="utf-8") as fh:
            _ENGLISH_WORDS = tuple(word.strip() for word in fh if word.strip())

    return _ENGLISH_WORDS
