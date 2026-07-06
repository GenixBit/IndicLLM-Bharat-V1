from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as HFTokenizersTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from bharat.tokenizer import BharatTokenizer, load_tokenizer


def _make_tiny_bpe_tokenizer() -> HFTokenizersTokenizer:
    """Build a tiny BPE tokenizer with ~500 tokens for offline testing."""
    bpe = BPE()
    tok = HFTokenizersTokenizer(bpe)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|instruction|>", "<|response|>"],
    )

    texts = [
        "Hello world how are you today",
        "I am fine thank you",
        "This is a test of the tokenizer",
        "नमस्ते भारत यह एक परीक्षण है",
        "System user assistant conversation",
        "The quick brown fox jumps over",
        "the lazy dog near the river bank",
        "What is the capital of France",
        "Paris is the capital of France",
        "Machine learning is fascinating",
        "Tokenization breaks text into pieces",
        "EOS PAD BOS UNK special markers",
        "Hello नमस्ते मशीन लर्निंग",
        "User: hi Assistant: hello there",
        "First Second Third Fourth Fifth",
        "a b c d e f g h i j k l m n o p",
        "q r s t u v w x y z",
    ]
    tok.train_from_iterator(texts, trainer=trainer)
    tok.add_special_tokens(["<|endoftext|>", "<|pad|>", "<|instruction|>", "<|response|>"])
    return tok


def _make_tiny_sp_model(tmp_dir: Path) -> Path:
    """Train a tiny SentencePiece model and return its path."""
    import sentencepiece as sp

    input_path = tmp_dir / "sp_input.txt"
    with open(input_path, "w") as f:
        for line in [
            "Hello world how are you today",
            "This is a test sentence for tokenizer",
            "नमस्ते भारत यह एक परीक्षण है",
            "Machine learning is very fascinating",
            "a b c d e f g h i j k l m n o p",
            "The quick brown fox jumps over the lazy dog",
            "Tokenization is the process of breaking text",
            "Each piece is called a token in NLP",
        ]:
            f.write(line + "\n")

    model_path = tmp_dir / "tiny_sp.model"
    sp.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(tmp_dir / "tiny_sp"),
        vocab_size=32,
        character_coverage=0.9995,
        model_type="unigram",
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
    )
    return model_path


def create_tiny_bpe_tokenizer() -> BharatTokenizer:
    """Create a tiny BPE tokenizer and wrap it. Saves to temp file."""
    bpe_tok = _make_tiny_bpe_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        bpe_tok.save(f.name)
        json_path = f.name
    return load_tokenizer(json_path)


def create_tiny_sp_tokenizer() -> BharatTokenizer:
    """Create a tiny SentencePiece tokenizer and wrap it."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _make_tiny_sp_model(Path(tmp))
        return load_tokenizer(str(model_path))


def get_tiny_bpe_tokenizer_path() -> str:
    """Return path to a tiny BPE tokenizer.json that can be loaded."""
    bpe_tok = _make_tiny_bpe_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        bpe_tok.save(f.name)
        return f.name


def get_tiny_sp_model_path() -> str:
    """Return path to a tiny SentencePiece .model file."""
    with tempfile.TemporaryDirectory() as tmp:
        model_path = _make_tiny_sp_model(Path(tmp))
        return str(model_path)
