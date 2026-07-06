from __future__ import annotations

import json
import tempfile

import pytest
import torch

from bharat.posttraining.dpo import DPOConfig, dpo_train
from bharat.posttraining.preference_dataset import PreferenceDataset, dpo_collate
from bharat.posttraining.templates import Template
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


class TestDPODataset:
    def test_dataset_initialization(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer
        )
        assert len(dataset) == 1

    def test_response_mask_correct_shape(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer
        )
        item = dataset[0]
        chosen = item["chosen_ids"]
        mask = item["chosen_response_mask"]
        # mask should be aligned to the full chosen sequence
        assert mask.shape[0] == chosen.shape[0]


class TestDPOTraining:
    def test_cpu_backward(self, preferences_jsonl, tokenizer):
        from train.pretrain import GPT, GPTConfig

        model_cfg = GPTConfig(
            vocab_size=50257,
            n_embd=32,
            n_head=4,
            n_layer=2,
            block_size=64,
            bias=False,
        )
        policy = GPT(model_cfg)
        ref = GPT(model_cfg)

        # Save initial params
        init_policy = {k: v.clone() for k, v in policy.named_parameters() if v.requires_grad}
        init_ref = {k: v.clone() for k, v in ref.named_parameters() if v.requires_grad}

        config = DPOConfig(
            data_path=preferences_jsonl,
            max_iters=1,
            batch_size=1,
            block_size=64,
            beta=0.1,
            device="cpu",
        )
        config.output_dir = tempfile.mkdtemp()

        loss = dpo_train(policy, ref, config, tokenizer)

        assert loss > 0, "DPO loss should be positive"

        # Policy parameters should have changed
        policy_changed = False
        for k, v in policy.named_parameters():
            if v.requires_grad and not torch.equal(v, init_policy[k]):
                policy_changed = True
                break
        assert policy_changed, "Policy model parameters should change after training"

        # Reference parameters should NOT have changed
        ref_changed = False
        for k, v in ref.named_parameters():
            if v.requires_grad and not torch.equal(v, init_ref[k]):
                ref_changed = True
                break
        assert not ref_changed, "Reference model parameters should NOT change"

        import shutil

        shutil.rmtree(config.output_dir, ignore_errors=True)

    def test_different_prompt_lengths_in_batch(self, tokenizer):
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

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        batch = [dataset[i] for i in range(2)]
        result = dpo_collate(batch, pad_token_id=0)

        assert result["chosen_ids"].shape[0] == 2
        assert result["chosen_response_mask"].shape[0] == 2

        # The two prompts should have different response mask patterns
        assert not torch.equal(result["chosen_response_mask"][0], result["chosen_response_mask"][1])

        import os

        os.unlink(path)

    def test_different_response_lengths(self, tokenizer):
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

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        item = dataset[0]

        # Chosen and rejected should have different lengths
        assert item["chosen_ids"].shape[0] != item["rejected_ids"].shape[0]

        import os

        os.unlink(path)

    def test_empty_assistant_response(self, tokenizer):
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

        dataset = PreferenceDataset(path, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        item = dataset[0]

        # Even with empty chosen response, the assistant prefix + suffix are present
        assert item["chosen_response_mask"] is not None
        assert item["rejected_response_mask"] is not None

        import os

        os.unlink(path)
