from alphabet_detector import AlphabetDetector

ad = AlphabetDetector()


# filter out non-latin tokens
def filter(token: str) -> bool:
    return ad.is_latin(token.lstrip("▁"))
