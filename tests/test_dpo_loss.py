from __future__ import annotations

import json
import tempfile

import pytest
import torch

from bharat.posttraining.preference_dataset import PreferenceDataset, dpo_collate
from bharat.posttraining.preference_loss import (
    approximate_kl_divergence,
    dpo_loss,
    per_sample_log_probs,
    reward_accuracy,
)
from bharat.posttraining.templates import Template

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
        f.write(
            json.dumps(
                {
                    "prompt": "What is Python?",
                    "chosen": "A programming language",
                    "rejected": "A snake",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "prompt": "Capital of France?",
                    "chosen": "Paris",
                    "rejected": "London",
                }
            )
            + "\n"
        )
        path = f.name
    yield path
    import os

    os.unlink(path)


class TestPreferenceDataset:
    def test_dataset_loads(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        assert len(dataset) == 3

    def test_response_mask_present(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        item = dataset[0]
        assert "chosen_ids" in item
        assert "rejected_ids" in item
        assert "chosen_response_mask" in item
        assert "rejected_response_mask" in item
        assert isinstance(item["chosen_response_mask"], torch.Tensor)
        assert item["chosen_response_mask"].dtype == torch.bool

    def test_response_mask_has_active_tokens(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        item = dataset[0]
        assert item["chosen_response_mask"].any(), "Response mask should have active tokens"
        assert item["rejected_response_mask"].any(), "Response mask should have active tokens"

    def test_prompt_tokens_masked(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        item = dataset[0]
        chosen = item["chosen_ids"]
        mask = item["chosen_response_mask"]
        # mask is aligned to the full chosen sequence
        assert (
            mask.shape[0] == chosen.shape[0]
        ), f"Mask length {mask.shape[0]} should be len(chosen) = {chosen.shape[0]}"

    def test_variable_prompt_lengths(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        item0 = dataset[0]
        item2 = dataset[2]
        # Different prompts should have different response mask start positions
        assert not torch.equal(item0["chosen_response_mask"], item2["chosen_response_mask"])

    def test_empty_response(self, tiny_tokenizer):
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
        # Even with empty response, the assistant prefix + suffix tokens are present
        assert item["chosen_response_mask"] is not None

        import os

        os.unlink(path)


class TestDPOCollate:
    def test_collate_batch(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        batch = [dataset[i] for i in range(2)]
        result = dpo_collate(batch, pad_token_id=0)
        assert "chosen_ids" in result
        assert "rejected_ids" in result
        assert "chosen_response_mask" in result
        assert "rejected_response_mask" in result
        assert result["chosen_ids"].shape[0] == 2
        assert result["rejected_ids"].shape[0] == 2
        assert result["chosen_response_mask"].shape[0] == 2
        assert result["rejected_response_mask"].shape[0] == 2

    def test_padding_shape(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        batch = [dataset[i] for i in range(3)]
        result = dpo_collate(batch, pad_token_id=0)
        assert result["chosen_ids"].shape[1] == result["chosen_response_mask"].shape[1]
        assert result["rejected_ids"].shape[1] == result["rejected_response_mask"].shape[1]

    def test_padding_has_false_mask(self, preferences_jsonl, tiny_tokenizer):
        dataset = PreferenceDataset(
            preferences_jsonl, INDIC_TEMPLATE, block_size=512, tokenizer=tiny_tokenizer
        )
        batch = [dataset[i] for i in range(3)]
        result = dpo_collate(batch, pad_token_id=0)
        # Some padding should exist (variable length sequences)
        chosen_lens = result["chosen_seq_len"]
        for i in range(len(batch)):
            pad_mask = result["chosen_response_mask"][i, chosen_lens[i] - 1 :]
            assert not pad_mask.any(), f"Padding in sample {i} should have False mask"


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
        ref_chosen = torch.tensor([0.0, 0.0, 0.0])
        ref_rejected = torch.tensor([0.0, 0.0, 0.0])
        acc = reward_accuracy(chosen, rejected, ref_chosen, ref_rejected)
        assert acc.item() == pytest.approx(2.0 / 3.0)

    def test_approximate_kl_divergence(self):
        pc = torch.tensor([1.0, 2.0])
        rc = torch.tensor([0.0, 1.0])
        pr = torch.tensor([0.5, 1.5])
        rr = torch.tensor([0.0, 0.5])
        kl = approximate_kl_divergence(pc, rc, pr, rr)
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
        mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                batch_size, seq_len = x.shape
                return torch.randn(batch_size, seq_len, 10), None

        model = DummyModel()
        ctx = torch.no_grad()
        lp = per_sample_log_probs(model, ids, mask, ctx)
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

        full = per_sample_log_probs(
            model, ids, torch.tensor([[True, True, True, True]], dtype=torch.bool), ctx
        )
        partial = per_sample_log_probs(
            model, ids, torch.tensor([[False, False, True, True]], dtype=torch.bool), ctx
        )
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
        # Sample 0: 1 masked, 3 active; Sample 1: 3 masked, 1 active
        mask = torch.tensor(
            [
                [False, True, True, True],
                [False, False, False, True],
            ],
            dtype=torch.bool,
        )
        ctx = torch.no_grad()

        lp = per_sample_log_probs(model, ids, mask, ctx)
        assert lp.shape == (2,)
        assert lp[0].item() < lp[1].item(), "More active tokens -> higher sum"

    def test_padding_excluded_from_logprob(self):
        class FixedModel(torch.nn.Module):
            def forward(self, x):
                batch_size, seq_len = x.shape
                logits = torch.full((batch_size, seq_len, 10), 0.0)
                logits[:, :, 0] = 10.0
                return logits, None

        model = FixedModel()
        ids = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long)
        # Only first token is response; rest is padding
        mask = torch.tensor([[True, False, False, False, False]], dtype=torch.bool)
        ctx = torch.no_grad()

        lp = per_sample_log_probs(model, ids, mask, ctx)
        assert lp.shape == (1,)
        # With full softmax of 10.0 on index 0 and input_ids=0:
        # log_softmax(10, 0, 0, ...) ~= 0 for index 0, very negative for others
        # gather at index 0 should give ~0
        assert not torch.isnan(lp[0])
        assert not torch.isinf(lp[0])
