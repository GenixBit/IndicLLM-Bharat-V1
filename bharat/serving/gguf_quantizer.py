"""Model exporter to quantized GGUF Q8_0 format for edge inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult
from bharat.serving.gguf_tensor_writer import write_gguf_q8_0_tensors
from bharat.tokenizer import BharatTokenizer


@dataclass
class GGUFQuantizationResult:
    output_path: Path
    file_size_bytes: int
    tensors_quantized: int


def export_model_to_gguf_q8_0(
    model: BharatForCausalLM,
    config: BharatModelConfig,
    _tokenizer: BharatTokenizer | None,
    output_path: str | Path,
) -> GGUFQuantizationResult:
    """Export BharatForCausalLM state dict to GGUF Q8_0 binary format."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Filter out RoPE caches and non-weight buffers
    state_dict = model.state_dict()
    quant_tensors: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if "inv_freq" in k or "cos_cached" in k or "sin_cached" in k:
            continue
        quant_tensors[k] = v.detach().cpu().to(torch.float32)

    metadata = (
        GGUFMetadataEntry(key="general.architecture", value_type="string", value="bharat"),
        GGUFMetadataEntry(
            key="general.name", value_type="string", value=f"bharat-{config.hidden_size}h"
        ),
        GGUFMetadataEntry(
            key="bharat.context_length", value_type="int", value=config.max_position_embeddings
        ),
        GGUFMetadataEntry(
            key="bharat.embedding_length", value_type="int", value=config.hidden_size
        ),
        GGUFMetadataEntry(
            key="bharat.block_count", value_type="int", value=config.num_hidden_layers
        ),
        GGUFMetadataEntry(
            key="bharat.feed_forward_length", value_type="int", value=config.intermediate_size
        ),
        GGUFMetadataEntry(
            key="bharat.attention.head_count", value_type="int", value=config.num_attention_heads
        ),
        GGUFMetadataEntry(
            key="bharat.attention.head_count_kv", value_type="int", value=config.num_key_value_heads
        ),
    )

    preflight = GGUFPreflightResult(
        schema_version=1,
        architecture="bharat",
        alignment=32,
        tensor_count=len(quant_tensors),
        output_file=out_p.name,
        metadata=metadata,
        gguf_tensor_type="q8_0",
    )

    write_res = write_gguf_q8_0_tensors(preflight, quant_tensors, out_p)

    return GGUFQuantizationResult(
        output_path=out_p,
        file_size_bytes=write_res.bytes_written,
        tensors_quantized=write_res.tensor_count,
    )
