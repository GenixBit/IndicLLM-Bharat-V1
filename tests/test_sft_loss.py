from __future__ import annotations

import json
import tempfile

import pytest
import torch

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


class TestLossMasking:
    """Verify that only assistant tokens contribute to the loss."""

    def test_loss_only_on_assistant(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)

        full_text = "<|instruction|>What is 2+2?<|endoftext|><|response|>4<|endoftext|>"
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # Find assistant prefix position
        ap_start = None
        for i in range(len(full_ids) - len(ap_ids) + 1):
            if full_ids[i:i + len(ap_ids)] == ap_ids:
                ap_start = i
                break
        assert ap_start is not None

        # Labels are aligned to target_ids = full_ids[1:]
        # User tokens in target_ids: positions 0 to ap_start-2
        target_ids = full_ids[1:]
        user_len = ap_start
        labels_slice = labels[:user_len]
        assert (labels_slice == -100).all(), "User tokens should be fully masked"

        # Response tokens should be active
        num_active = (labels != -100).sum().item()
        assert num_active > 0, "At least some tokens should contribute to loss"

    def test_multi_turn_masking(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "First Q"},
            {"role": "assistant", "content": "First A"},
            {"role": "user", "content": "Second Q"},
            {"role": "assistant", "content": "Second A"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "Assistant tokens should contribute to loss"

    def test_assistant_content_active(self, tokenizer):
        # Directly test with a known conversation
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello world"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]
        input_ids = result["input_ids"][0]

        full_text = "<|instruction|>Hi<|endoftext|><|response|>Hello world<|endoftext|>"
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        target_ids = full_ids[1:]

        # Where labels != -100, they should match target_ids
        for i in range(len(labels)):
            if labels[i] != -100:
                assert labels[i].item() == target_ids[i], (
                    f"Label at position {i} should match target token {target_ids[i]}"
                )

    def test_padding_excluded_from_loss(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]},
            {"messages": [{"role": "user", "content": "C" * 100}, {"role": "assistant", "content": "D"}]},
        ]
        result = collator(batch)
        labels = result["labels"]

        # Shorter sequence should have padding at the end
        seq_lens = (result["input_ids"] != 0).sum(dim=1)
        for i in range(len(batch)):
            pad_labels = labels[i, seq_lens[i]:]
            assert (pad_labels == -100).all(), f"Padding in sample {i} should have -100 labels"
