from alphabet_detector import AlphabetDetector

ad = AlphabetDetector()


def filter(token: str) -> bool:
    return ad.is_latin(token.lstrip("▁"))
