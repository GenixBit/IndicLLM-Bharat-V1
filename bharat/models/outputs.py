from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bharat.models.cache import PastKeyValues


@dataclass
class BharatModelOutput:
    """
    Output of ``BharatModel.forward``.

    Attributes:
        last_hidden_state: Token hidden states of shape
            ``(batch_size, sequence_length, hidden_size)``.
        past_key_values: KV caches for each decoder layer, or ``None``
            when ``use_cache=False``.
    """

    last_hidden_state: Any  # torch.Tensor
    past_key_values: PastKeyValues | None = None


@dataclass
class BharatCausalLMOutput:
    """
    Output of ``BharatForCausalLM.forward``.

    Attributes:
        logits: Prediction scores of shape
            ``(batch_size, sequence_length, vocab_size)``.
        loss: Cross-entropy next-token prediction loss, or ``None``
            when labels are not supplied.
        past_key_values: KV caches for each decoder layer, or ``None``
            when ``use_cache=False``.
    """

    logits: Any  # torch.Tensor
    loss: Any | None = None  # torch.Tensor | None
    past_key_values: PastKeyValues | None = None
