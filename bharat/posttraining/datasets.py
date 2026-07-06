from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from bharat.tokenizer import BharatTokenizer


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        template: Any,
        block_size: int,
    ) -> None:
        self.block_size = block_size
        self.template = template
        self.samples: list[list[dict[str, str]]] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if "messages" in item:
                    messages = item["messages"]
                elif "instruction" in item and "response" in item:
                    messages = [
                        {"role": "user", "content": item["instruction"]},
                        {
                            "role": "assistant",
                            "content": item.get("response", item.get("output", "")),
                        },
                    ]
                else:
                    continue
                self.samples.append(messages)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"messages": self.samples[idx]}


class SFTTokenizedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        jsonl_path: str | Path,
        template: Any,
        block_size: int,
        tokenizer: BharatTokenizer,
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
        self.system_prefix_ids: list[int] = (
            tokenizer.encode(template.system_prefix, add_special_tokens=False)
            if template.system_prefix
            else []
        )
        self.suffix_ids: list[int] = tokenizer.encode(template.suffix, add_special_tokens=False)
        self.samples: list[list[dict[str, str]]] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                messages: list[dict[str, str]] = []
                if "messages" in item:
                    messages = item["messages"]
                elif "instruction" in item:
                    messages = [
                        {"role": "user", "content": item["instruction"]},
                        {
                            "role": "assistant",
                            "content": item.get("response", item.get("output", "")),
                        },
                    ]
                else:
                    continue
                self.samples.append(messages)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_segment_ranges(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[int], list[tuple[int, int, str]]]:
        full_ids: list[int] = []
        segments: list[tuple[int, int, str]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                if self.template.system_prefix:
                    prefix_ids = self.system_prefix_ids
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                    suffix_ids = self.suffix_ids
                    start = len(full_ids)
                    full_ids.extend(prefix_ids + content_ids + suffix_ids)
                    end = len(full_ids)
                    segments.append((start, end, "system"))
            elif role == "user":
                prefix_ids = self.user_prefix_ids
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = self.suffix_ids
                start = len(full_ids)
                full_ids.extend(prefix_ids + content_ids + suffix_ids)
                end = len(full_ids)
                segments.append((start, end, "user"))
            elif role == "assistant":
                prefix_ids = self.assistant_prefix_ids
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = self.suffix_ids
                start = len(full_ids)
                full_ids.extend(prefix_ids + content_ids + suffix_ids)
                end = len(full_ids)
                ap_start = start
                ap_end = start + len(prefix_ids)
                content_start = ap_end
                content_end = content_start + len(content_ids)
                segments.append((ap_start, ap_end, "assistant_prefix"))
                segments.append((content_start, content_end, "assistant_content"))
                if suffix_ids:
                    segments.append((content_end, end, "assistant_suffix"))

        return full_ids, segments

    def _build_labels(
        self, full_ids: list[int], segments: list[tuple[int, int, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_ids = full_ids[: self.block_size + 1]

        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_ids[1:], dtype=torch.long)
        labels = torch.full_like(input_ids, -100)

        for start, end, seg_type in segments:
            if seg_type == "assistant_content" or seg_type == "assistant_suffix":
                t_start = max(0, start - 1)
                t_end = min(len(labels), end - 1)
                if t_start < t_end:
                    labels[t_start:t_end] = target_ids[t_start:t_end]

        return input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.samples[idx]
        full_ids, segments = self._get_segment_ranges(messages)
        input_ids, labels = self._build_labels(full_ids, segments)

        if len(input_ids) < self.block_size:
            pad_len = self.block_size - len(input_ids)
            pad = torch.full((pad_len,), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "labels": labels}
