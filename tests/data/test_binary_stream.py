from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bharat.data.binary_stream import (
    BinaryShardHeader,
    BinaryTokenPacker,
    MMapTokenDataset,
    MMapTokenShard,
    MMapTokenStream,
    pack_text_corpus,
)
from bharat.tokenizer import load_tokenizer
from scripts.pack_tokens import main as pack_tokens_main
from scripts.pack_tokens import parse_args


class TestBinaryStream:
    def test_header_serialization(self):
        header = BinaryShardHeader(
            version=1,
            dtype="uint16",
            vocab_size=64000,
            num_tokens=1000,
            num_documents=10,
        )
        d = header.to_dict()
        assert d["dtype"] == "uint16"
        assert d["vocab_size"] == 64000

        loaded = BinaryShardHeader.from_dict(d)
        assert loaded.num_tokens == 1000
        assert loaded.num_documents == 10

    def test_binary_token_packer_and_mmap_dataset(self, tmp_path: Path):
        out_dir = tmp_path / "shards"
        packer = BinaryTokenPacker(
            output_dir=out_dir,
            prefix="test_shard",
            vocab_size=500,
            dtype="uint16",
            max_tokens_per_shard=50,
        )

        # Write 3 small documents (total 90 tokens -> creates 2 shards of 60 and 30 tokens)
        packer.add_document(list(range(30)))
        packer.add_document(list(range(30)))
        packer.add_document(list(range(30)))
        shards = packer.close()

        assert len(shards) == 2
        assert shards[0].is_file()
        assert shards[1].is_file()

        # Test MMapTokenShard
        s0 = MMapTokenShard(shards[0])
        assert len(s0) == 60
        assert s0.header.dtype == "uint16"
        assert int(s0[0]) == 0
        assert int(s0[29]) == 29

        # Test MMapTokenDataset
        ds = MMapTokenDataset(shards, block_size=16)
        assert len(ds) == (90 - 1) // 16  # 5 samples
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert len(x) == 16
        assert len(y) == 16
        # Autoregressive shift: y[i] == x[i+1] (for sequential tokens)
        assert torch.equal(y[:-1], x[1:])

    def test_mmap_token_stream_iterable(self, tmp_path: Path):
        out_dir = tmp_path / "shards_stream"
        packer = BinaryTokenPacker(
            output_dir=out_dir,
            prefix="stream_shard",
            vocab_size=1000,
            dtype="uint16",
            max_tokens_per_shard=100,
        )
        packer.add_document(list(range(250)))
        shards = packer.close()

        stream = MMapTokenStream(shards, block_size=32, shuffle=False)
        loader = DataLoader(stream, batch_size=2)

        batches = list(loader)
        assert len(batches) > 0
        bx, by = batches[0]
        assert bx.shape == (2, 32)
        assert by.shape == (2, 32)

    def test_pack_text_corpus(self, tmp_path: Path):
        text_file = tmp_path / "corpus.txt"
        text_file.write_text(
            "Hello India!\nIndicLLM Bharat is awesome.\nNamaste.", encoding="utf-8"
        )

        tok = load_tokenizer("gpt2")
        out_dir = tmp_path / "packed"

        shards = pack_text_corpus(
            tokenizer=tok,
            input_file=text_file,
            output_dir=out_dir,
            prefix="corpus_shard",
            max_tokens_per_shard=1000,
        )

        assert len(shards) == 1
        assert shards[0].is_file()

        ds = MMapTokenDataset(shards, block_size=4)
        assert len(ds) > 0

    def test_cli_pack_tokens_main(self, tmp_path: Path):
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps({"text": "Artificial Intelligence in India."})
            + "\n"
            + json.dumps({"text": "Machine Learning and Neural Networks."})
            + "\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "cli_shards"

        args = parse_args(
            [
                "--input",
                str(jsonl_file),
                "--output-dir",
                str(out_dir),
                "--prefix",
                "cli_shard",
            ]
        )
        assert args.input == str(jsonl_file)

        code = pack_tokens_main(
            [
                "--input",
                str(jsonl_file),
                "--output-dir",
                str(out_dir),
                "--prefix",
                "cli_shard",
            ]
        )
        assert code == 0
        assert len(list(out_dir.glob("*.bin"))) > 0
