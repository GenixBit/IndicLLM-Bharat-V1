from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from bharat.posttraining.templates import Template
from bharat.tokenizer.base import BharatTokenizer


@dataclass
class SFTCollator:
    tokenizer: BharatTokenizer
    template: Template
    block_size: int
    pad_token_id: int = 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for item in batch:
            if "messages" in item:
                input_ids, labels = self._process_messages(item["messages"])
            elif "text" in item:
                input_ids, labels = self._process_text(item["text"])
            else:
                continue

            if input_ids.numel() == 0:
                continue

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        if not all_input_ids:
            return {
                "input_ids": torch.zeros((0, 0), dtype=torch.long),
                "labels": torch.zeros((0, 0), dtype=torch.long),
            }

        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "labels": labels}

    def _get_role_prefix_ids(self) -> dict[str, list[int]]:
        prefixes: dict[str, list[int]] = {}
        prefixes["user"] = self.tokenizer.encode(
            self.template.user_prefix, add_special_tokens=False
        )
        prefixes["assistant"] = self.tokenizer.encode(
            self.template.assistant_prefix, add_special_tokens=False
        )
        if self.template.system_prefix:
            prefixes["system"] = self.tokenizer.encode(
                self.template.system_prefix, add_special_tokens=False
            )
        prefixes["suffix"] = self.tokenizer.encode(self.template.suffix, add_special_tokens=False)
        return prefixes

    def _segment_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[int], list[tuple[int, int, str]]]:
        prefixes = self._get_role_prefix_ids()
        full_ids: list[int] = []
        segments: list[tuple[int, int, str]] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prefix_ids = prefixes.get("system", [])
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = prefixes["suffix"]
                start = len(full_ids)
                full_ids.extend(prefix_ids + content_ids + suffix_ids)
                segments.append((start, len(full_ids), "system"))

            elif role == "user":
                prefix_ids = prefixes["user"]
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = prefixes["suffix"]
                start = len(full_ids)
                full_ids.extend(prefix_ids + content_ids + suffix_ids)
                segments.append((start, len(full_ids), "user"))

            elif role == "assistant":
                prefix_ids = prefixes["assistant"]
                content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                suffix_ids = prefixes["suffix"]
                start = len(full_ids)

                ap_start = len(full_ids)
                full_ids.extend(prefix_ids)
                ap_end = len(full_ids)

                cont_start = len(full_ids)
                full_ids.extend(content_ids)
                cont_end = len(full_ids)

                full_ids.extend(suffix_ids)
                end = len(full_ids)

                segments.append((ap_start, ap_end, "assistant_prefix"))
                segments.append((cont_start, cont_end, "assistant_content"))
                if suffix_ids:
                    segments.append((cont_end, end, "assistant_suffix"))

        return full_ids, segments

    def _process_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_ids, segments = self._segment_messages(messages)

        if len(full_ids) < 2:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

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

    def _process_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: self.block_size + 1]

        if len(ids) < 2:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        ap_ids = self.tokenizer.encode(self.template.assistant_prefix, add_special_tokens=False)
        up_ids = self.tokenizer.encode(self.template.user_prefix, add_special_tokens=False)
        sp_ids = (
            self.tokenizer.encode(self.template.system_prefix, add_special_tokens=False)
            if self.template.system_prefix
            else []
        )
        suffix_ids = self.tokenizer.encode(self.template.suffix, add_special_tokens=False)

        all_marker_ids = set(ap_ids + up_ids + sp_ids + suffix_ids)

        full = torch.tensor(ids, dtype=torch.long)
        input_ids = full[:-1]
        target_ids = full[1:]
        labels = torch.full_like(input_ids, -100)

        ap_len = len(ap_ids)

        # Find all assistant prefix positions in full_ids
        ap_positions: list[int] = []
        for i in range(len(ids) - ap_len + 1):
            if ids[i : i + ap_len] == ap_ids:
                ap_positions.append(i)

        for ap_start in ap_positions:
            response_start = ap_start + ap_len

            # Find the next role marker after response_start
            response_end = len(ids)
            for j in range(response_start, len(ids)):
                if ids[j] in all_marker_ids:
                    # Check if this starts a known prefix
                    remaining = len(ids) - j
                    if remaining >= len(up_ids) and ids[j : j + len(up_ids)] == up_ids:
                        response_end = j
                        break
                    if sp_ids and remaining >= len(sp_ids) and ids[j : j + len(sp_ids)] == sp_ids:
                        response_end = j
                        break
                    if remaining >= len(ap_ids) and ids[j : j + len(ap_ids)] == ap_ids:
                        response_end = j
                        break

            # Map to target_ids positions (shifted by 1)
            t_start = max(0, response_start - 1)
            t_end = min(len(labels), response_end - 1)
            if t_start < t_end:
                labels[t_start:t_end] = target_ids[t_start:t_end]

        return input_ids, labels
