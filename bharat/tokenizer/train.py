from __future__ import annotations

from pathlib import Path

from bharat.tokenizer.base import BharatTokenizer
from bharat.tokenizer.loader import _SentencePieceWrapper


def train_bpe_tokenizer(
    texts: list[str],
    vocab_size: int = 64000,
    special_tokens: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> BharatTokenizer:
    """Train a BPE tokenizer on the given texts.

    Args:
        texts: List of text strings to train on.
        vocab_size: Target vocabulary size.
        special_tokens: Additional special tokens to add.
        output_dir: Optional directory to save the trained tokenizer.

    Returns:
        A BharatTokenizer instance wrapping the trained tokenizer.
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|pad|>"]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path / "tokenizer.json"))

    return _SentencePieceWrapper(tokenizer)


def train_sentencepiece_tokenizer(
    texts: list[str],
    vocab_size: int = 64000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    output_dir: str | Path | None = None,
) -> BharatTokenizer:
    """Train a SentencePiece tokenizer on the given texts.

    Args:
        texts: List of text strings to train on.
        vocab_size: Target vocabulary size.
        model_type: SentencePiece model type ('bpe' or 'unigram').
        character_coverage: Coverage of characters to include.
        output_dir: Optional directory to save the trained tokenizer.

    Returns:
        A BharatTokenizer instance wrapping the trained tokenizer.
    """
    import tempfile

    import sentencepiece as spm

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for t in texts:
            f.write(t + "\n")
        temp_path = f.name

    model_prefix = "bharat_tokenizer"
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    import os

    os.unlink(temp_path)

    from tokenizers import SentencePieceBPETokenizer

    tok = SentencePieceBPETokenizer.from_file(f"{model_prefix}.model")
    result = _SentencePieceWrapper(tok)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy(f"{model_prefix}.model", output_path / "tokenizer.model")
        shutil.copy(f"{model_prefix}.vocab", output_path / "tokenizer.vocab")

    os.unlink(f"{model_prefix}.model")
    os.unlink(f"{model_prefix}.vocab")

    return result
