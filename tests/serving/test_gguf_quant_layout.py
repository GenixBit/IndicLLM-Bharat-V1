import pytest

from bharat.serving.gguf_quant_layout import (
    GGML_TYPE_Q8_0,
    build_q8_0_tensor_descriptors,
    q8_0_payload_size,
)
from bharat.serving.gguf_writer import GGUFTensorInventoryEntry


def test_q8_0_payload_size_uses_34_bytes_per_32_values() -> None:
    assert q8_0_payload_size((32,)) == 34
    assert q8_0_payload_size((2, 32)) == 68


def test_q8_0_payload_size_rejects_partial_blocks() -> None:
    with pytest.raises(ValueError, match="multiple of 32"):
        q8_0_payload_size((33,))


def test_descriptors_are_sorted_and_aligned() -> None:
    descriptors = build_q8_0_tensor_descriptors(
        (
            GGUFTensorInventoryEntry(name="z", shape=(64,)),
            GGUFTensorInventoryEntry(name="a", shape=(32,)),
        ),
        alignment=32,
    )

    assert [descriptor.name for descriptor in descriptors] == ["a", "z"]
    assert [descriptor.ggml_type for descriptor in descriptors] == [GGML_TYPE_Q8_0] * 2
    assert descriptors[0].offset == 0
    assert descriptors[0].payload_bytes == 34
    assert descriptors[1].offset == 64
    assert descriptors[1].payload_bytes == 68


def test_layout_is_independent_of_input_order() -> None:
    first = build_q8_0_tensor_descriptors(
        (
            GGUFTensorInventoryEntry(name="b", shape=(32,)),
            GGUFTensorInventoryEntry(name="a", shape=(32,)),
        ),
        alignment=32,
    )
    second = build_q8_0_tensor_descriptors(
        reversed(
            (
                GGUFTensorInventoryEntry(name="b", shape=(32,)),
                GGUFTensorInventoryEntry(name="a", shape=(32,)),
            )
        ),
        alignment=32,
    )

    assert first == second


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate tensor names"):
        build_q8_0_tensor_descriptors(
            (
                GGUFTensorInventoryEntry(name="same", shape=(32,)),
                GGUFTensorInventoryEntry(name="same", shape=(64,)),
            ),
            alignment=32,
        )


def test_invalid_alignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        build_q8_0_tensor_descriptors(
            (GGUFTensorInventoryEntry(name="tensor", shape=(32,)),),
            alignment=24,
        )
