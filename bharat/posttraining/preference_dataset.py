from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from bharat.posttraining.templates import Template, format_conversation


class PreferenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        template: Template,
        block_size: int,
        tokenizer: object,
    ) -> None:
        self.block_size = block_size
        self.template = template
        self.tokenizer = tokenizer
        self.assistant_prefix_ids: list[int] = tokenizer.encode(
            template.assistant_prefix, add_special_tokens=False
        )
        self.user_prefix_ids: list[int] = tokenizer.encode(
            template.user_prefix, add_special_tokens=False
        )
        self.suffix_ids: list[int] = tokenizer.encode(
            template.suffix, add_special_tokens=False
        )

        self.samples: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                prompt = item.get("prompt", item.get("instruction", ""))
                chosen = item.get("chosen", item.get("response", ""))
                rejected = item.get("rejected", "")
                self.samples.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _build_sequence(
        self, prompt: str, response: str
    ) -> tuple[list[int], list[int], int, int]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = format_conversation(self.template, messages)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:self.block_size]

        ap_len = len(self.assistant_prefix_ids)

        # Find assistant prefix position
        prompt_end = 0
        for i in range(len(ids) - ap_len + 1):
            if ids[i:i + ap_len] == self.assistant_prefix_ids:
                prompt_end = i + ap_len
                break

        if prompt_end == 0:
            prompt_end = len(ids)

        # Build response mask for target positions (shifted by 1)
        # target_ids = ids[1:], so target position i corresponds to ids[i+1]
        # Response tokens start at prompt_end in ids, so in target_ids they start at prompt_end - 1
        response_start = prompt_end - 1  # first response token in target_ids
        response_end = len(ids) - 1  # last valid target position

        return ids, response_start, response_end, len(ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        chosen_ids, chosen_rs, chosen_re, chosen_len = self._build_sequence(
            sample["prompt"], sample["chosen"]
        )
        rejected_ids, rejected_rs, rejected_re, rejected_len = self._build_sequence(
            sample["prompt"], sample["rejected"]
        )

        chosen_ids_t = torch.tensor(chosen_ids, dtype=torch.long)
        rejected_ids_t = torch.tensor(rejected_ids, dtype=torch.long)

        # Response masks aligned to target positions (input_ids[:, 1:])
        # For chosen: target length = len(chosen_ids) - 1 or just len for full
        chosen_seq_len = len(chosen_ids)
        rejected_seq_len = len(rejected_ids)

        chosen_mask = torch.zeros(chosen_seq_len, dtype=torch.bool)
        rejected_mask = torch.zeros(rejected_seq_len, dtype=torch.bool)

        if chosen_rs < chosen_re:
            chosen_mask[chosen_rs:chosen_re] = True

        if rejected_rs < rejected_re:
            rejected_mask[rejected_rs:rejected_re] = True

        return {
            "chosen_ids": chosen_ids_t,
            "rejected_ids": rejected_ids_t,
            "chosen_response_mask": chosen_mask,
            "rejected_response_mask": rejected_mask,
            "chosen_seq_len": torch.tensor(chosen_seq_len, dtype=torch.long),
            "rejected_seq_len": torch.tensor(rejected_seq_len, dtype=torch.long),
        }


def dpo_collate(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    chosen = [item["chosen_ids"] for item in batch]
    rejected = [item["rejected_ids"] for item in batch]
    chosen_masks = [item["chosen_response_mask"] for item in batch]
    rejected_masks = [item["rejected_response_mask"] for item in batch]
    chosen_lens = torch.stack([item["chosen_seq_len"] for item in batch])
    rejected_lens = torch.stack([item["rejected_seq_len"] for item in batch])

    chosen_padded = pad_sequence(chosen, batch_first=True, padding_value=pad_token_id)
    rejected_padded = pad_sequence(rejected, batch_first=True, padding_value=pad_token_id)

    chosen_mask_padded = pad_sequence(chosen_masks, batch_first=True, padding_value=False)
    rejected_mask_padded = pad_sequence(rejected_masks, batch_first=True, padding_value=False)

    return {
        "chosen_ids": chosen_padded,
        "rejected_ids": rejected_padded,
        "chosen_response_mask": chosen_mask_padded,
        "rejected_response_mask": rejected_mask_padded,
        "chosen_seq_len": chosen_lens,
        "rejected_seq_len": rejected_lens,
    }
