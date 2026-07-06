from __future__ import annotations

import json
import tempfile

import pytest
import torch

from bharat.posttraining.preference_dataset import PreferenceDataset, dpo_collate
from bharat.posttraining.preference_loss import (
    dpo_loss,
    kl_divergence,
    per_sample_log_probs,
    reward_accuracy,
)
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
        f.write(json.dumps({
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
        }) + "\n")
        f.write(json.dumps({
            "prompt": "What is Python?",
            "chosen": "A programming language",
            "rejected": "A snake",
        }) + "\n")
        f.write(json.dumps({
            "prompt": "Capital of France?",
            "chosen": "Paris",
            "rejected": "London",
        }) + "\n")
        path = f.name
    yield path
    import os
    os.unlink(path)


class TestPreferenceDataset:
    def test_dataset_loads(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        assert len(dataset) == 3

    def test_per_sample_prompt_end(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        item = dataset[0]
        assert "chosen_ids" in item
        assert "rejected_ids" in item
        assert "chosen_prompt_end" in item
        assert "rejected_prompt_end" in item
        assert item["chosen_prompt_end"].item() > 0
        assert item["rejected_prompt_end"].item() > 0

    def test_variable_length_prompts(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        item0 = dataset[0]
        item2 = dataset[2]

        assert item0["chosen_prompt_end"].item() != item2["chosen_prompt_end"].item()


class TestDPOCollate:
    def test_collate_batch(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        batch = [dataset[i] for i in range(2)]
        result = dpo_collate(batch, pad_token_id=0)
        assert "chosen_ids" in result
        assert "rejected_ids" in result
        assert "chosen_prompt_end" in result
        assert "rejected_prompt_end" in result
        assert result["chosen_ids"].shape[0] == 2
        assert result["rejected_ids"].shape[0] == 2
        assert result["chosen_prompt_end"].shape[0] == 2
        assert result["rejected_prompt_end"].shape[0] == 2

    def test_padding_shape(self, preferences_jsonl, tokenizer):
        dataset = PreferenceDataset(preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tokenizer)
        batch = [dataset[i] for i in range(3)]
        result = dpo_collate(batch, pad_token_id=0)
        assert result["chosen_ids"].shape[1] == result["rejected_ids"].shape[1]
        assert result["chosen_ids"].shape[1] <= 512


class TestDPOLossFunctions:
    def test_dpo_loss_smoke(self):
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([0.0, 1.0, 2.0])
        ref_chosen = torch.tensor([0.5, 1.5, 2.5])
        ref_rejected = torch.tensor([0.0, 0.5, 1.0])
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        assert loss.item() > 0
        assert isinstance(loss.item(), float)

    def test_dpo_loss_prefers_correct(self):
        chosen = torch.tensor([100.0])
        rejected = torch.tensor([0.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=1.0)
        assert loss.item() < 1.0

    def test_dpo_loss_penalizes_wrong(self):
        chosen = torch.tensor([0.0])
        rejected = torch.tensor([100.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=1.0)
        assert loss.item() > 0.5

    def test_reward_accuracy(self):
        chosen = torch.tensor([2.0, 1.0, 3.0])
        rejected = torch.tensor([1.0, 2.0, 1.0])
        acc = reward_accuracy(chosen, rejected)
        assert acc.item() == pytest.approx(2.0 / 3.0)

    def test_kl_divergence(self):
        pc = torch.tensor([1.0, 2.0])
        rc = torch.tensor([0.0, 1.0])
        pr = torch.tensor([0.5, 1.5])
        rr = torch.tensor([0.0, 0.5])
        kl = kl_divergence(pc, rc, pr, rr)
        assert kl.item() >= 0

    def test_dpo_loss_zero_beta(self):
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([0.0, 1.0])
        ref_chosen = torch.tensor([0.5, 1.5])
        ref_rejected = torch.tensor([0.0, 0.5])
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.0)
        assert loss.item() > 0

    def test_dpo_loss_high_beta(self):
        chosen = torch.tensor([1.0])
        rejected = torch.tensor([0.0])
        ref_chosen = torch.tensor([0.5])
        ref_rejected = torch.tensor([0.0])
        loss_low = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
        loss_high = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=10.0)
        assert loss_high.item() < loss_low.item()


class TestPerSampleLogProbs:
    def test_no_prompt_equals_full_sequence(self):
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        prompt_lengths = torch.tensor([0])

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                batch_size, seq_len = x.shape
                return torch.randn(batch_size, seq_len, 10), None

        model = DummyModel()
        ctx = torch.no_grad()
        lp = per_sample_log_probs(model, ids, prompt_lengths, ctx)
        assert lp.shape == (1,)

    def test_masking_reduces_prob(self):
        class FixedModel(torch.nn.Module):
            def forward(self, x):
                batch_size, seq_len = x.shape
                logits = torch.full((batch_size, seq_len, 10), 0.0)
                logits[:, :, 0] = 10.0
                return logits, None

        model = FixedModel()
        ids = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        ctx = torch.no_grad()

        full = per_sample_log_probs(model, ids, torch.tensor([0]), ctx)
        partial = per_sample_log_probs(model, ids, torch.tensor([3]), ctx)
        assert partial.item() > full.item()

    def test_variable_prompts_batch(self):
        class FixedModel(torch.nn.Module):
            def forward(self, x):
                batch_size, seq_len = x.shape
                logits = torch.full((batch_size, seq_len, 10), 0.0)
                logits[:, :, 0] = 10.0
                return logits, None

        model = FixedModel()
        ids = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.long)
        prompt_lengths = torch.tensor([1, 4])
        ctx = torch.no_grad()

        lp = per_sample_log_probs(model, ids, prompt_lengths, ctx)
        assert lp.shape == (2,)
        assert lp[0].item() < lp[1].item()
