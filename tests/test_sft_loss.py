from __future__ import annotations

import json
import tempfile

import pytest

from bharat.posttraining.collators import SFTCollator
from bharat.posttraining.datasets import SFTDataset
from bharat.posttraining.templates import Template, format_conversation
from bharat.tokenizer import load_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("gpt2")


INDIC_TEMPLATE = Template(
    name="indic_instruction",
    system_prefix="",
    user_prefix="<|instruction|>",
    assistant_prefix="<|response|>",
    suffix="<|endoftext|>",
)


@pytest.fixture
def sft_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"instruction": "What is 2+2?", "response": "4"}) + "\n")
        f.write(json.dumps({"instruction": "What is Python?", "response": "A programming language"}) + "\n")
        path = f.name
    yield path
    import os
    os.unlink(path)


class TestTemplateFormatting:
    def test_indic_template(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = format_conversation(INDIC_TEMPLATE, messages)
        assert "<|instruction|>Hello<|endoftext|>" in text
        assert "<|response|>Hi there<|endoftext|>" in text

    def test_system_role(self):
        template = Template(
            name="test",
            system_prefix="<|system|>",
            user_prefix="<|user|>",
            assistant_prefix="<|assistant|>",
            suffix="<|end|>",
        )
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        text = format_conversation(template, messages)
        assert "<|system|>Be helpful<|end|>" in text
        assert "<|user|>Hello<|end|>" in text
        assert "<|assistant|>Hi<|end|>" in text

    def test_tool_role(self):
        template = Template(
            name="test",
            system_prefix="",
            user_prefix="<|user|>",
            assistant_prefix="<|assistant|>",
            suffix="<|end|>",
        )
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "Sunny"},
        ]
        text = format_conversation(template, messages)
        assert "<|user|>Weather?<|end|>" in text
        assert "<|assistant|>Sunny<|end|>" in text


class TestSFTDataset:
    def test_dataset_loads(self, sft_jsonl, tokenizer):
        dataset = SFTDataset(sft_jsonl, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 2

    def test_dataset_messages_format(self, tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }) + "\n")
            path = f.name

        dataset = SFTDataset(path, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 1
        item = dataset[0]
        assert "<|response|>Hi" in item["text"]

        import os
        os.unlink(path)


class TestSFTCollator:
    def test_assistant_loss_active(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"text": "<|instruction|>What is 2+2?<|endoftext|><|response|>4<|endoftext|>"},
        ]
        result = collator(batch)
        labels = result["labels"]
        decoded_labels = labels[0].tolist()
        non_masked = sum(1 for v in decoded_labels if v != -100)
        assert non_masked > 0, "Assistant tokens should not be masked"

    def test_non_assistant_tokens_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"text": "<|instruction|>Hello<|endoftext|><|response|>World<|endoftext|>"},
        ]
        result = collator(batch)
        labels = result["labels"][0]

        ap_len = len(tokenizer.encode("<|response|>", add_special_tokens=False))
        ap_pos = None
        ids = tokenizer.encode("<|instruction|>Hello<|endoftext|><|response|>World<|endoftext|>",
                                add_special_tokens=False)
        for i in range(len(ids) - ap_len + 1):
            if ids[i:i + ap_len] == tokenizer.encode("<|response|>", add_special_tokens=False):
                ap_pos = i
                break

        assert ap_pos is not None

        nz_before = (labels[:ap_pos] != -100).sum().item()
        assert nz_before == 0, "Tokens before assistant prefix should be masked"

        nz_after = (labels[ap_pos:] != -100).sum().item()
        assert nz_after > 0, "Tokens from assistant prefix onward should contribute to loss"

    def test_padding_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"text": "<|instruction|>A<|endoftext|><|response|>B<|endoftext|>"},
            {"text": "<|instruction|>C<|endoftext|><|response|>D<|endoftext|>"},
        ]
        result = collator(batch)
        labels = result["labels"]
        padding_mask = labels == -100
        assert padding_mask.any(), "Should have some padding"

    def test_empty_batch(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        result = collator([])
        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape[0] == 0

    def test_variable_length_batch(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        texts = [
            {"text": "<|instruction|>Short<|endoftext|><|response|>A<|endoftext|>"},
            {"text": "<|instruction|>" + "very " * 50 + "long<|endoftext|><|response|>B<|endoftext|>"},
        ]
        result = collator(texts)
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["labels"].shape[1]
        assert result["input_ids"].shape[1] <= 512 + 1


class TestLossMasking:
    """Verify that only assistant tokens contribute to the loss."""

    def test_loss_only_on_assistant(self, tokenizer):
        user_text = "What is 2+2?"
        assistant_text = "4"
        full_text = f"<|instruction|>{user_text}<|endoftext|><|response|>{assistant_text}<|endoftext|>"

        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [{"text": full_text}]
        result = collator(batch)
        labels = result["labels"][0]

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        ap_len = len(ap_ids)

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        ap_pos = None
        for i in range(len(full_ids) - ap_len + 1):
            if full_ids[i:i + ap_len] == ap_ids:
                ap_pos = i
                break

        assert ap_pos is not None

        tokens_before = full_ids[:ap_pos]

        for i, tid in enumerate(tokens_before):
            if i < len(labels):
                assert labels[i].item() == -100, f"Token at position {i} ({tid}) should be masked"

        num_active = (labels != -100).sum().item()
        assert num_active > 0, "At least some tokens should contribute to loss"

    def test_multi_turn_masking(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        text = ("<|instruction|>First Q<|endoftext|><|response|>First A<|endoftext|>"
                "<|instruction|>Second Q<|endoftext|><|response|>Second A<|endoftext|>")
        batch = [{"text": text}]
        result = collator(batch)
        labels = result["labels"][0]
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "Assistant tokens should contribute to loss"
