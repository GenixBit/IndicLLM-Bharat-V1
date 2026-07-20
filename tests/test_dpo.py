from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from bharat.posttraining.dpo import DPOConfig, DPOResult, dpo_train
from bharat.posttraining.preference_dataset import PreferenceDataset, dpo_collate
from bharat.posttraining.templates import Template
from train.pretrain import GPT, GPTConfig

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

INDIC_TEMPLATE = Template(
    name="indic_instruction",
    system_prefix="",
    user_prefix="<|instruction|>",
    assistant_prefix="<|response|>",
    suffix="<|endoftext|>",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model(vocab_size: int = 256):
    return GPT(
        GPTConfig(vocab_size=vocab_size, n_embd=32, n_head=4, n_layer=2, block_size=64, bias=False)
    )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestDPOConfig:
    def test_valid_config(self):
        c = DPOConfig(max_iters=1, batch_size=1, block_size=64, save_interval=1, log_interval=1)
        assert c.max_iters == 1

    def test_max_iters_zero_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            DPOConfig(max_iters=0)

    def test_max_iters_negative_raises(self):
        with pytest.raises(ValueError, match="max_iters"):
            DPOConfig(max_iters=-1)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            DPOConfig(max_iters=1, batch_size=0)

    def test_block_size_one_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            DPOConfig(max_iters=1, block_size=1)

    def test_lr_zero_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DPOConfig(max_iters=1, learning_rate=0)

    def test_lr_negative_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DPOConfig(max_iters=1, learning_rate=-1)

    def test_save_interval_zero_raises(self):
        with pytest.raises(ValueError, match="save_interval"):
            DPOConfig(max_iters=1, save_interval=0)

    def test_log_interval_zero_raises(self):
        with pytest.raises(ValueError, match="log_interval"):
            DPOConfig(max_iters=1, log_interval=0)

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            DPOConfig(max_iters=1, device="invalid_device")


# ---------------------------------------------------------------------------
# Offline tokenizer fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def preferences_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "prompt": "What is 2+2?",
                    "chosen": "4",
                    "rejected": "5",
                }
            )
            + "\n"
        )
        path = f.name
    yield path
    import os

    os.unlink(path)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestDPODataset:
    def test_dataset_initialization(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        assert len(dataset) == 1

    def test_response_mask_correct_shape(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        item = dataset[0]
        chosen = item["chosen_ids"]
        mask = item["chosen_response_mask"]
        assert mask.shape[0] == chosen.shape[0]


# ---------------------------------------------------------------------------
# Training tests (offline, use tiny_tokenizer)
# ---------------------------------------------------------------------------


class TestDPOTraining:
    def test_empty_dataset_raises(self, tiny_tokenizer):
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            data_path = f.name

        config = DPOConfig(
            data_path=data_path, max_iters=1, batch_size=1, block_size=64, device="cpu"
        )
        config.output_dir = tempfile.mkdtemp()

        with pytest.raises(ValueError, match="empty"):
            dpo_train(policy, ref, config, tiny_tokenizer)

        import os

        os.unlink(data_path)

    def test_one_sample_batch_four(self, tiny_tokenizer):
        """1 preference sample with batch_size=4 must still train."""
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "What is 2+2?",
                        "chosen": "4",
                        "rejected": "5",
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = DPOConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=4,
            block_size=64,
            beta=0.1,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = dpo_train(policy, ref, config, tiny_tokenizer)
        assert isinstance(result, DPOResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0
        assert result.completed_steps == 1
        assert result.next_step == 1

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_cpu_backward(self, tiny_tokenizer):
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        init_policy = {k: v.clone() for k, v in policy.named_parameters() if v.requires_grad}
        init_ref = {k: v.clone() for k, v in ref.named_parameters() if v.requires_grad}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "What is 2+2?",
                        "chosen": "4",
                        "rejected": "5",
                    }
                )
                + "\n"
            )
            data_path = f.name

        config = DPOConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            beta=0.1,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = dpo_train(policy, ref, config, tiny_tokenizer)
        assert isinstance(result, DPOResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0
        assert result.best_loss > 0
        assert result.completed_steps == 1
        assert result.next_step == 1

        policy_changed = False
        for k, v in policy.named_parameters():
            if v.requires_grad and not torch.equal(v, init_policy[k]):
                policy_changed = True
                break
        assert policy_changed, "Policy model parameters should change after training"

        ref_changed = False
        for k, v in ref.named_parameters():
            if v.requires_grad and not torch.equal(v, init_ref[k]):
                ref_changed = True
                break
        assert not ref_changed, "Reference model parameters should NOT change"

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_different_prompt_lengths_in_batch(self, tiny_tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Short",
                        "chosen": "Good answer",
                        "rejected": "Bad answer",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "prompt": "A very long prompt that should produce different token lengths",
                        "chosen": "Great response here",
                        "rejected": "Poor response here",
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer)
        batch = [dataset[i] for i in range(2)]
        result = dpo_collate(batch, pad_token_id=0)

        assert result["chosen_ids"].shape[0] == 2
        assert result["chosen_response_mask"].shape[0] == 2
        assert not torch.equal(result["chosen_response_mask"][0], result["chosen_response_mask"][1])

        import os

        os.unlink(path)

    def test_different_response_lengths(self, tiny_tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Same prompt",
                        "chosen": "Short response",
                        "rejected": "This is a much longer response to the same prompt",
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer)
        item = dataset[0]

        assert item["chosen_ids"].shape[0] != item["rejected_ids"].shape[0]

        import os

        os.unlink(path)

    def test_empty_chosen_response(self, tiny_tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Test",
                        "chosen": "",
                        "rejected": "Something",
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer)
        item = dataset[0]

        assert item["chosen_response_mask"] is not None
        assert item["rejected_response_mask"] is not None

        import os

        os.unlink(path)

    def test_empty_rejected_response(self, tiny_tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Test",
                        "chosen": "Something",
                        "rejected": "",
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer)
        item = dataset[0]

        assert item["chosen_response_mask"] is not None
        assert item["rejected_response_mask"] is not None

        import os

        os.unlink(path)

    def test_both_responses_empty(self, tiny_tokenizer):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Test",
                        "chosen": "",
                        "rejected": "",
                    }
                )
                + "\n"
            )
            path = f.name

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer)
        item = dataset[0]

        assert item["chosen_response_mask"] is not None
        assert item["rejected_response_mask"] is not None

        import os

        os.unlink(path)

    def test_both_empty_responses_training(self, tiny_tokenizer):
        """Both responses empty — prefix tokens still provide non-zero mask."""
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "Test", "chosen": "", "rejected": ""}) + "\n")
            path = f.name

        config = DPOConfig(
            data_path=path,
            max_iters=1,
            batch_size=4,
            block_size=64,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        result = dpo_train(policy, ref, config, tiny_tokenizer)
        assert isinstance(result, DPOResult)
        assert math.isfinite(result.final_loss)
        assert result.final_loss > 0
        assert result.next_step == 1

        import os

        os.unlink(path)

    # --- Best checkpoint tests ---

    def test_best_pt_created(self, tiny_tokenizer):
        """best.pt is created during training."""
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "What is 2+2?",
                        "chosen": "4",
                        "rejected": "5",
                    }
                )
                + "\n"
            )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = DPOConfig(
            data_path=data_path,
            max_iters=3,
            batch_size=1,
            block_size=64,
            beta=0.1,
            learning_rate=1e-3,
            device="cpu",
            save_interval=5,
        )
        config.output_dir = out_dir

        result = dpo_train(policy, ref, config, tiny_tokenizer)
        assert isinstance(result, DPOResult)

        best_path = out_dir / "best.pt"
        assert best_path.exists(), "best.pt should be created"

        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert math.isfinite(best_ckpt["best_loss"])
        assert best_ckpt["best_loss"] <= best_ckpt["final_loss"]
        assert best_ckpt["completed_steps"] > 0
        assert best_ckpt["next_step"] == best_ckpt["completed_steps"]
        assert best_ckpt["metadata"]["tokenizer_hash"]

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)

    def test_best_loss_matches_checkpoint(self, tiny_tokenizer):
        """best_loss in DPOResult matches the saved checkpoint."""
        policy = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)
        ref = _make_tiny_model(vocab_size=tiny_tokenizer.vocab_size)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "What is 2+2?",
                        "chosen": "4",
                        "rejected": "5",
                    }
                )
                + "\n"
            )
            data_path = f.name

        out_dir = Path(tempfile.mkdtemp())
        config = DPOConfig(
            data_path=data_path,
            max_iters=1,
            batch_size=1,
            block_size=64,
            beta=0.1,
            device="cpu",
        )
        config.output_dir = out_dir

        result = dpo_train(policy, ref, config, tiny_tokenizer)

        best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        assert abs(result.best_loss - best_ckpt["best_loss"]) < 1e-6
        assert result.next_step == best_ckpt["next_step"]

        import os
        import shutil

        os.unlink(data_path)
        shutil.rmtree(out_dir, ignore_errors=True)
