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
            text = item["text"]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[:self.block_size + 1]

            if len(ids) < 2:
                continue

            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels = torch.full_like(input_ids, -100)

            ap_ids = self.tokenizer.encode(
                self.template.assistant_prefix, add_special_tokens=False
            )
            ap_len = len(ap_ids)

            for i in range(len(input_ids) - ap_len + 1):
                if torch.equal(input_ids[i:i + ap_len], torch.tensor(ap_ids)):
                    start = i + ap_len
                    labels[start:] = input_ids[start:]
                    break

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        if not all_input_ids:
            return {"input_ids": torch.zeros((0, 0), dtype=torch.long),
                    "labels": torch.zeros((0, 0), dtype=torch.long)}

        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)

        return {"input_ids": input_ids, "labels": labels}
