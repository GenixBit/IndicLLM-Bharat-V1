from __future__ import annotations

import json
import tempfile

import pytest
import torch

from bharat.posttraining.collators import SFTCollator
from bharat.posttraining.datasets import SFTDataset
from bharat.posttraining.sft import SFTConfig, SFTResult, sft_train
from bharat.posttraining.templates import Template
from bharat.tokenizer import load_tokenizer
from train.pretrain import GPT, GPTConfig


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
        f.write(
            json.dumps({"instruction": "What is Python?", "response": "A programming language"})
            + "\n"
        )
        path = f.name
    yield path
    import os

    os.unlink(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model():
    return GPT(GPTConfig(vocab_size=50257, n_embd=32, n_head=4, n_layer=2, block_size=64, bias=False))


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestSFTDataset:
    def test_dataset_initialization(self, sft_jsonl, tokenizer):
        dataset = SFTDataset(sft_jsonl, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 2

    def test_dataset_returns_messages(self, sft_jsonl, tokenizer):
        dataset = SFTDataset(sft_jsonl, INDIC_TEMPLATE, block_size=512)
        item = dataset[0]
        assert "messages" in item
        assert len(item["messages"]) == 2
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"

    def test_dataset_messages_format(self, tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = SFTDataset(path, INDIC_TEMPLATE, block_size=512)
        assert len(dataset) == 1
        item = dataset[0]
        assert item["messages"][0]["content"] == "Hello"
        assert item["messages"][1]["content"] == "Hi"

        import os

        os.unlink(path)


# ---------------------------------------------------------------------------
# Loss masking tests
# ---------------------------------------------------------------------------


class TestSFTLossMasking:
    def test_assistant_only_loss(self, tokenizer):
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
        non_masked = (labels != -100).sum().item()
        assert non_masked > 0, "Assistant tokens should not be masked"

    def test_user_tokens_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        ap_len = len(ap_ids)

        full_ids = tokenizer.encode(
            "<|instruction|>Hello<|endoftext|><|response|>World<|endoftext|>",
            add_special_tokens=False,
        )
        ap_start = None
        for i in range(len(full_ids) - ap_len + 1):
            if full_ids[i : i + ap_len] == ap_ids:
                ap_start = i
                break

        assert ap_start is not None

        user_end_in_targets = ap_start - 1
        user_labels = labels[: max(0, user_end_in_targets)]
        assert (user_labels == -100).all(), "User tokens should be fully masked"

    def test_system_tokens_masked(self, tokenizer):
        template_with_system = Template(
            name="test_system",
            system_prefix="<|system|>",
            user_prefix="<|user|>",
            assistant_prefix="<|assistant|>",
            suffix="<|end|>",
        )
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=template_with_system,
            block_size=512,
        )
        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        sys_ids = tokenizer.encode("<|system|>Be concise<|end|>", add_special_tokens=False)
        user_start = len(sys_ids)

        sys_end_in_targets = user_start - 1
        sys_labels = labels[: max(0, sys_end_in_targets)]
        assert (sys_labels == -100).all(), "System tokens should be fully masked"

    def test_padding_masked(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        batch = [
            {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]},
            {"messages": [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]},
        ]
        result = collator(batch)
        labels = result["labels"]
        padding_mask = labels == -100
        assert padding_mask.any(), "Should have some padding"


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestSFTCollator:
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
        batch = [
            {
                "messages": [
                    {"role": "user", "content": "Short"},
                    {"role": "assistant", "content": "A"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "very " * 50 + "long"},
                    {"role": "assistant", "content": "B"},
                ]
            },
        ]
        result = collator(batch)
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["labels"].shape[1]

    def test_first_response_token_included(self, tokenizer):
        collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=512,
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        batch = [{"messages": messages}]
        result = collator(batch)
        labels = result["labels"][0]

        full_ids = tokenizer.encode(
            "<|instruction|>Hi<|endoftext|><|response|>Hello<|endoftext|>",
            add_special_tokens=False,
        )
        target_ids = full_ids[1:]
        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        ap_len = len(ap_ids)

        ap_start = None
        for i in range(len(full_ids) - ap_len + 1):
            if full_ids[i : i + ap_len] == ap_ids:
                ap_start = i
                break

        non_masked_mask = labels != -100
        # First response token in target_ids = ap_start + ap_len - 1
        first_response_idx = ap_start + ap_len - 1
        assert first_response_idx < len(labels)
        assert non_masked_mask[first_response_idx].item(), (
            f"First response token at index {first_response_idx} should not be masked"
        )
        assert labels[first_response_idx].item() == target_ids[first_response_idx]

    def test_user_tokens_masked_multi_turn(self, tokenizer):
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

        ap_ids = tokenizer.encode("<|response|>", add_special_tokens=False)
        up_ids = tokenizer.encode("<|instruction|>", add_special_tokens=False)

        full_text = (
            "<|instruction|>First Q<|endoftext|><|response|>First A<|endoftext|>"
            "<|instruction|>Second Q<|endoftext|><|response|>Second A<|endoftext|>"
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        ap_positions = []
        for i in range(len(full_ids) - len(ap_ids) + 1):
            if full_ids[i : i + len(ap_ids)] == ap_ids:
                ap_positions.append(i)

        up_positions = []
        for i in range(len(full_ids) - len(up_ids) + 1):
            if full_ids[i : i + len(up_ids)] == up_ids:
                up_positions.append(i)

        for up_start in up_positions:
            up_end = len(full_ids)
            for p in sorted(ap_positions + up_positions):
                if p > up_start:
                    up_end = p
                    break

            target_start = max(0, up_start - 1)
            target_end = min(len(labels), up_end - 1)
            if target_start < target_end:
                user_label_slice = labels[target_start:target_end]
                assert (user_label_slice == -100).all(), (
                    f"User tokens at target positions {target_start}:{target_end} should be masked"
                )

    def test_multi_turn_assistant_active(self, tokenizer):
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

        transitions = 0
        prev_masked = True
        for v in labels:
            if prev_masked and v != -100:
                transitions += 1
            prev_masked = v == -100

        assert transitions >= 2, f"Expected at least 2 active regions, got {transitions}"

    def test_truncated_conversation(self, tokenizer):
        small_collator = SFTCollator(
            tokenizer=tokenizer,
            template=INDIC_TEMPLATE,
            block_size=10,
        )
        messages = [
            {"role": "user", "content": "This is a very long user message that will be truncated"},
            {"role": "assistant", "content": "Short"},
        ]
        batch = [{"messages": messages}]
        result = small_collator(batch)
        labels = result["labels"][0]

        # Make sure no label is NaN
        assert not torch.isnan(labels.float()).any()


# ---------------------------------------------------------------------------
# End-to-end training tests
# ---------------------------------------------------------------------------


class TestSFTOneStepTraining:
    def test_empty_dataset_raises(self):
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            data_path = f.name

        config = SFTConfig(
            data_path=data_path, max_iters=1, batch_size=1, block_size=64, device="cpu"
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="empty"):
            sft_train(model, config)

        import os

        os.unlink(data_path)

    def test_dataset_smaller_than_batch(self, tokenizer):
        """1 sample with batch_size=2 must still train successfully."""
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=2,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = sft_train(model, config, tokenizer)
        assert isinstance(result, SFTResult)
        assert result.final_loss > 0
        assert result.completed_steps == 1

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_max_iters_one(self, tokenizer):
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = sft_train(model, config, tokenizer)
        assert isinstance(result, SFTResult)
        assert result.final_loss > 0
        assert result.best_loss > 0
        assert result.completed_steps == 1
        assert result.samples_processed >= 1
        assert result.active_tokens >= 1

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_cpu_backward(self, tokenizer):
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            learning_rate=1e-3,
            warmup_iters=0,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        init_params = {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}

        result = sft_train(model, config, tokenizer)
        assert isinstance(result, SFTResult)
        assert result.final_loss > 0

        params_changed = False
        for k, v in model.named_parameters():
            if v.requires_grad and not torch.equal(v, init_params[k]):
                params_changed = True
                break
        assert params_changed, "At least one model parameter should change after training"

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_sample_without_assistant_response(self, tokenizer):
        """Sample with no assistant turn should be rejected by the collator."""
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Only user message"}
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="zero active assistant"):
            sft_train(model, config, tokenizer)

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_sample_truncated_before_assistant(self, tokenizer):
        """Very small block_size may truncate before assistant content."""
        model = _make_tiny_model()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "A" * 100},
                            {"role": "assistant", "content": "B"},
                        ]
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = SFTConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=10,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="zero active assistant"):
            sft_train(model, config, tokenizer)

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)
