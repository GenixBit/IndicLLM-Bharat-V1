from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from bharat.posttraining.templates import Template, format_conversation


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        template: Template,
        block_size: int,
    ) -> None:
        self.block_size = block_size
        self.template = template
        self.samples: list[str] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if "messages" in item:
                    text = format_conversation(template, item["messages"])
                elif "instruction" in item and "response" in item:
                    messages = [
                        {"role": "user", "content": item["instruction"]},
                        {"role": "assistant", "content": item.get("response", item.get("output", ""))},
                    ]
                    text = format_conversation(template, messages)
                else:
                    continue
                self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.samples[idx]
        return {"text": text}


class SFTTokenizedDataset(Dataset[dict[str, torch.Tensor]]):
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
        self.samples: list[list[int]] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                messages: list[dict[str, str]] = []
                if "messages" in item:
                    messages = item["messages"]
                elif "instruction" in item:
                    messages = [
                        {"role": "user", "content": item["instruction"]},
                        {"role": "assistant", "content": item.get("response", item.get("output", ""))},
                    ]
                else:
                    continue
                text = format_conversation(template, messages)
                ids = tokenizer.encode(text, add_special_tokens=False)
                self.samples.append(ids)

    def __len__(self) -> int:
        return len(self.samples)

    def _find_assistant_range(self, ids: list[int]) -> tuple[int, int]:
        ap_len = len(self.assistant_prefix_ids)
        for i in range(len(ids) - ap_len + 1):
            if ids[i:i + ap_len] == self.assistant_prefix_ids:
                start = i + ap_len
                return start, len(ids)
        return 0, 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.samples[idx][:self.block_size + 1]
        if len(ids) < self.block_size + 1:
            ids = ids + [0] * (self.block_size + 1 - len(ids))

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        astart, aend = self._find_assistant_range(ids[:-1])
        if astart < aend:
            labels[astart:aend] = input_ids[astart:aend]

        return {"input_ids": input_ids, "labels": labels}
