from __future__ import annotations

import json
from pathlib import Path

import torch
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

    def _encode_pair(
        self, prompt: str, response: str
    ) -> tuple[list[int], int]:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = format_conversation(self.template, messages)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:self.block_size]

        ap_len = len(self.assistant_prefix_ids)
        prompt_end = 0
        for i in range(len(ids) - ap_len + 1):
            if ids[i:i + ap_len] == self.assistant_prefix_ids:
                prompt_end = i + ap_len
                break

        if prompt_end == 0:
            prompt_end = len(ids)

        return ids, prompt_end

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        chosen_ids, chosen_prompt_end = self._encode_pair(
            sample["prompt"], sample["chosen"]
        )
        rejected_ids, rejected_prompt_end = self._encode_pair(
            sample["prompt"], sample["rejected"]
        )

        chosen_ids = torch.tensor(chosen_ids, dtype=torch.long)
        rejected_ids = torch.tensor(rejected_ids, dtype=torch.long)

        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_prompt_end": torch.tensor(chosen_prompt_end, dtype=torch.long),
            "rejected_prompt_end": torch.tensor(rejected_prompt_end, dtype=torch.long),
        }


def dpo_collate(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    chosen = [item["chosen_ids"] for item in batch]
    rejected = [item["rejected_ids"] for item in batch]
    chosen_pe = torch.stack([item["chosen_prompt_end"] for item in batch])
    rejected_pe = torch.stack([item["rejected_prompt_end"] for item in batch])

    from torch.nn.utils.rnn import pad_sequence

    chosen_padded = pad_sequence(chosen, batch_first=True, padding_value=pad_token_id)
    rejected_padded = pad_sequence(rejected, batch_first=True, padding_value=pad_token_id)

    return {
        "chosen_ids": chosen_padded,
        "rejected_ids": rejected_padded,
        "chosen_prompt_end": chosen_pe,
        "rejected_prompt_end": rejected_pe,
    }
