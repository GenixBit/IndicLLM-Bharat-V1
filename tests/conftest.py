from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import yaml
from tokenizers import Tokenizer as HFTokenizersTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return ROOT / "configs"


@pytest.fixture
def gpt2_10m_config(configs_dir: Path) -> dict:
    with open(configs_dir / "gpt2-10m.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def gpt2_124m_config(configs_dir: Path) -> dict:
    with open(configs_dir / "gpt2-124m.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Offline tiny tokenizer fixtures (no internet required)
# ---------------------------------------------------------------------------


def _build_tiny_bpe_tokenizer() -> HFTokenizersTokenizer:
    """Build a ~300-vocab BPE tokenizer for offline tests."""
    bpe = BPE()
    tok = HFTokenizersTokenizer(bpe)
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|instruction|>", "<|response|>"],
    )
    tok.train_from_iterator(
        [
            "Hello world how are you today",
            "I am fine thank you",
            "System user assistant conversation",
            "What is the capital of France",
            "Paris is the capital of France",
            "Machine learning is fascinating",
            "Tokenization breaks text into pieces",
            "EOS PAD BOS UNK special markers",
            "नमस्ते भारत यह एक परीक्षण है",
            "Hello नमस्ते मशीन लर्निंग",
            "User: hi Assistant: hello there",
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        ],
        trainer=trainer,
    )
    return tok


@pytest.fixture(scope="session")
def tiny_bpe_tokenizer_json() -> str:
    """Return path to a pre-built tiny BPE tokenizer.json (valid for session)."""
    tok = _build_tiny_bpe_tokenizer()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tok.save(f.name)
        return f.name


@pytest.fixture(scope="session")
def tiny_sp_model_path() -> Path:
    """Train a tiny SentencePiece model once per session and return its path."""
    import sentencepiece as sp

    tmp = Path(tempfile.mkdtemp())
    input_path = tmp / "sp_input.txt"
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
    sp.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=str(tmp / "tiny_sp"),
        vocab_size=64,
        character_coverage=0.9995,
        model_type="unigram",
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
    )
    return tmp / "tiny_sp.model"


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_bpe_tokenizer_json: str):
    """Load the tiny BPE tokenizer as a BharatTokenizer (offline)."""
    from bharat.tokenizer import load_tokenizer

    return load_tokenizer(tiny_bpe_tokenizer_json)


@pytest.fixture(scope="session")
def tiny_sp_tokenizer(tiny_sp_model_path: Path):
    """Load the tiny SentencePiece tokenizer as a BharatTokenizer (offline)."""
    from bharat.tokenizer import load_tokenizer

    return load_tokenizer(str(tiny_sp_model_path))


# ---------------------------------------------------------------------------
# Fake GPT-2 tokenizer fixture for tests that need GPT-2 specific properties
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fake_gpt2_tokenizer():
    """Return a BharatTokenizer that mimics GPT-2 properties (offline-safe)."""
    from bharat.tokenizer.base import BharatTokenizer

    class _FakeGPT2(BharatTokenizer):
        @property
        def vocab_size(self) -> int:
            return 50257

        @property
        def eos_token_id(self) -> int:
            return 50256

        @property
        def pad_token_id(self) -> int:
            return 50256

        @property
        def tokenizer_type(self) -> str:
            return "gpt2"

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [hash(text) % self.vocab_size]

        def encode_batch(
            self, texts: list[str], add_special_tokens: bool = True
        ) -> list[list[int]]:
            return [[hash(t) % self.vocab_size] for t in texts]

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            return " ".join(str(i) for i in ids)

        def decode_batch(
            self, batch: list[list[int]], skip_special_tokens: bool = True
        ) -> list[str]:
            return [self.decode(ids) for ids in batch]

        def get_metadata(self) -> dict:
            return {
                "tokenizer_type": self.tokenizer_type,
                "vocab_size": self.vocab_size,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            }

        def fingerprint(self) -> str:
            return hashlib.sha256(b"fake_gpt2").hexdigest()

    return _FakeGPT2()
