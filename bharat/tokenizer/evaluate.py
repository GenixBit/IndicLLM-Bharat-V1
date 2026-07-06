from __future__ import annotations

from collections import Counter

from bharat.tokenizer.base import BharatTokenizer


def compression_ratio(tokenizer: BharatTokenizer, texts: list[str]) -> float:
    """Compute the compression ratio (characters / tokens).

    Higher is better — means fewer tokens per character.
    """
    total_chars = sum(len(t) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    if total_tokens == 0:
        return 0.0
    return total_chars / total_tokens


def fertility(tokenizer: BharatTokenizer, texts: list[str]) -> float:
    """Compute token fertility — average number of tokens per word.

    Lower is better — means tokens capture whole words.
    """
    total_words = sum(len(t.split()) for t in texts)
    total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
    if total_words == 0:
        return 0.0
    return total_tokens / total_words


def _decode_carefully(tokenizer: BharatTokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids)
    except Exception:
        return ""


def top_k_rare_tokens(
    tokenizer: BharatTokenizer, texts: list[str], k: int = 20
) -> list[tuple[str, int]]:
    """Return the K least frequently used tokens and their counts."""
    token_freq: Counter[int] = Counter()
    for text in texts:
        ids = tokenizer.encode(text)
        token_freq.update(ids)

    rare_tokens = [
        (_decode_carefully(tokenizer, [tid]), count)
        for tid, count in token_freq.most_common()[:-k - 1:-1]
    ]
    return rare_tokens


def top_k_common_tokens(
    tokenizer: BharatTokenizer, texts: list[str], k: int = 20
) -> list[tuple[str, int]]:
    """Return the K most frequently used tokens and their counts."""
    token_freq: Counter[int] = Counter()
    for text in texts:
        ids = tokenizer.encode(text)
        token_freq.update(ids)

    common_tokens = [
        (_decode_carefully(tokenizer, [tid]), count)
        for tid, count in token_freq.most_common(k)
    ]
    return common_tokens


def language_wise_fertility(
    tokenizer: BharatTokenizer,
    texts_by_lang: dict[str, list[str]],
) -> dict[str, float]:
    """Compute fertility per language to identify problematic languages."""
    return {
        lang: fertility(tokenizer, texts)
        for lang, texts in texts_by_lang.items()
    }


def code_efficiency(
    tokenizer: BharatTokenizer,
    code_snippets: list[str],
) -> float:
    """Compute compression ratio specifically for code snippets."""
    return compression_ratio(tokenizer, code_snippets)


def all_metrics(
    tokenizer: BharatTokenizer,
    texts: list[str],
) -> dict[str, float]:
    """Return all tokenizer evaluation metrics in a single dict."""
    return {
        "compression_ratio": compression_ratio(tokenizer, texts),
        "fertility": fertility(tokenizer, texts),
        "code_efficiency": code_efficiency(tokenizer, code_snippets=texts),
    }
