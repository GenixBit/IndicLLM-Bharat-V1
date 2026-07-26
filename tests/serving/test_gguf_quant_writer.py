import math

import pytest

from bharat.serving.gguf_quant_writer import (
    BLOCK_Q8_0_SIZE,
    QK8_0,
    _float16_to_float32,
    _float32_to_float16,
    dequantize_q8_0,
    quantize_q8_0,
)


def _zero_block() -> list[float]:
    return [0.0] * QK8_0


def _uniform_block(value: float) -> list[float]:
    return [value] * QK8_0


def _ramp_block() -> list[float]:
    return [float(i) for i in range(QK8_0)]


def _mixed_block() -> list[float]:
    return [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 127.0, -128.0] * (QK8_0 // 8)


class TestFloat16Conversion:
    def test_zero(self) -> None:
        bits = _float32_to_float16(0.0)
        assert bits == 0
        assert _float16_to_float32(0) == 0.0

    def test_negative_zero(self) -> None:
        bits = _float32_to_float16(-0.0)
        assert bits == 0x8000
        assert _float16_to_float32(0x8000) == 0.0

    def test_one(self) -> None:
        bits = _float32_to_float16(1.0)
        assert bits == 0x3C00
        assert _float16_to_float32(bits) == 1.0

    def test_negative_one(self) -> None:
        bits = _float32_to_float16(-1.0)
        assert bits == 0xBC00
        assert _float16_to_float32(bits) == -1.0

    def test_two(self) -> None:
        bits = _float32_to_float16(2.0)
        assert bits == 0x4000
        assert _float16_to_float32(bits) == 2.0

    def test_round_trip(self) -> None:
        for v in [0.0, 1.0, -1.0, 0.5, -0.5, 3.14, -3.14, 127.0, 0.007874]:
            bits = _float32_to_float16(v)
            restored = _float16_to_float32(bits)
            assert restored == pytest.approx(v, rel=1e-3)

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN|Inf"):
            _float32_to_float16(float("nan"))

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN|Inf"):
            _float32_to_float16(float("inf"))


class TestQuantize:
    def test_empty_data_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            quantize_q8_0([])

    def test_non_divisible_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of"):
            quantize_q8_0([1.0] * 10)

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            quantize_q8_0([1.0] * QK8_0 + [float("nan")] + [1.0] * (QK8_0 - 1))

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="Inf"):
            quantize_q8_0([1.0] * QK8_0 + [float("inf")] + [1.0] * (QK8_0 - 1))

    def test_zero_block(self) -> None:
        data = _zero_block()
        quantized = quantize_q8_0(data)
        assert len(quantized) == BLOCK_Q8_0_SIZE
        d_bits = quantized[0] | (quantized[1] << 8)
        assert d_bits == 0
        for j in range(QK8_0):
            assert quantized[2 + j] == 0

    def test_uniform_positive_block(self) -> None:
        value = 1.0
        data = _uniform_block(value)
        quantized = quantize_q8_0(data)
        assert len(quantized) == BLOCK_Q8_0_SIZE
        d_bits = quantized[0] | (quantized[1] << 8)
        d = _float16_to_float32(d_bits)
        assert d == pytest.approx(1.0 / 127.0, abs=1e-5)
        for j in range(QK8_0):
            assert quantized[2 + j] == 127

    def test_uniform_negative_block(self) -> None:
        data = _uniform_block(-1.0)
        quantized = quantize_q8_0(data)
        assert len(quantized) == BLOCK_Q8_0_SIZE
        for j in range(QK8_0):
            q = quantized[2 + j]
            q_signed = q - 256 if q >= 128 else q
            assert q_signed == -127

    def test_block_size_correct(self) -> None:
        data = _mixed_block()
        quantized = quantize_q8_0(data)
        assert len(quantized) == BLOCK_Q8_0_SIZE

    def test_multi_block_length(self) -> None:
        data = _mixed_block() * 3
        quantized = quantize_q8_0(data)
        expected = 3 * BLOCK_Q8_0_SIZE
        assert len(quantized) == expected

    def test_deterministic(self) -> None:
        data = _mixed_block() * 5
        first = quantize_q8_0(data)
        second = quantize_q8_0(list(data))
        assert first == second

    def test_reverse_data_changes_output(self) -> None:
        data = _mixed_block()
        forward = quantize_q8_0(data)
        reverse = quantize_q8_0(tuple(reversed(data)))
        rev_quantized = quantize_q8_0(list(reversed(data)))
        assert forward != reverse
        assert rev_quantized == reverse


class TestDequantize:
    def test_positive_count_required(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            dequantize_q8_0(b"", 0)

    def test_element_count_must_be_multiple(self) -> None:
        with pytest.raises(ValueError, match="multiple of"):
            dequantize_q8_0(b"", QK8_0 + 1)

    def test_data_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            dequantize_q8_0(b"\x00" * BLOCK_Q8_0_SIZE, QK8_0 * 2)


class TestRoundTrip:
    def test_zero(self) -> None:
        data = _zero_block()
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            assert restored == pytest.approx(original, abs=1e-6)

    def test_uniform_positive(self) -> None:
        data = _uniform_block(1.0)
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for restored in dequantized:
            assert restored == pytest.approx(1.0, abs=1.0 / 127.0)

    def test_uniform_negative(self) -> None:
        data = _uniform_block(-1.0)
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for restored in dequantized:
            assert restored == pytest.approx(-1.0, abs=1.0 / 127.0)

    def test_mixed_values(self) -> None:
        data = _mixed_block()
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            d = max(abs(original), 1e-8) * 0.02 + 0.6
            assert restored == pytest.approx(original, abs=d)

    def test_ramp_values(self) -> None:
        data = _ramp_block()
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            d = max(abs(original), 1e-8) * 0.02 + 0.15
            assert restored == pytest.approx(original, abs=d)

    def test_large_values(self) -> None:
        data = [v * 100.0 for v in _mixed_block()]
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            d = max(abs(original), 1e-8) * 0.02 + 60.0
            assert restored == pytest.approx(original, abs=d)

    def test_multi_block(self) -> None:
        data = _mixed_block() * 7
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            d = max(abs(original), 1e-8) * 0.02 + 0.6
            assert restored == pytest.approx(original, abs=d)

    def test_max_relative_error_nonzero_values(self) -> None:
        data = _mixed_block() * 7
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        errors = []
        for original, restored in zip(data, dequantized, strict=True):
            if abs(original) > 1.0:
                errors.append(abs(restored - original) / abs(original))
        assert errors
        max_rel = max(errors)
        assert max_rel < 0.02

    def test_cosine_similarity(self) -> None:
        data = _mixed_block() * 7
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        dot = sum(a * b for a, b in zip(data, dequantized, strict=True))
        norm_a = math.sqrt(sum(v * v for v in data))
        norm_b = math.sqrt(sum(v * v for v in dequantized))
        cos_sim = dot / (norm_a * norm_b)
        assert cos_sim > 0.999

    def test_deterministic_round_trip(self) -> None:
        data = _mixed_block() * 11
        q1 = quantize_q8_0(data)
        d1 = dequantize_q8_0(q1, len(data))
        q2 = quantize_q8_0(list(data))
        d2 = dequantize_q8_0(q2, len(data))
        assert q1 == q2
        for a, b in zip(d1, d2, strict=True):
            assert a == pytest.approx(b, abs=1e-6)

    def test_small_positive_values_underflow_to_zero(self) -> None:
        data = [0.001] * QK8_0
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for restored in dequantized:
            assert restored == 0.0

    def test_small_negative_values_underflow_to_zero(self) -> None:
        data = [-0.001] * QK8_0
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for restored in dequantized:
            assert restored == 0.0

    def test_values_above_fp16_underflow_threshold(self) -> None:
        data = [0.01] * QK8_0
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for restored in dequantized:
            assert restored > 0

    def test_alternating_sign(self) -> None:
        data = [float(1 if i % 2 == 0 else -1) for i in range(QK8_0)]
        quantized = quantize_q8_0(data)
        dequantized = dequantize_q8_0(quantized, len(data))
        for original, restored in zip(data, dequantized, strict=True):
            assert (restored > 0) == (original > 0)
            assert restored == pytest.approx(original, abs=1.0 / 127.0)


class TestByteLayout:
    def test_scale_in_first_two_bytes(self) -> None:
        data = _mixed_block()
        quantized = quantize_q8_0(data)
        d_bits = quantized[0] | (quantized[1] << 8)
        d = _float16_to_float32(d_bits)
        assert d > 0

    def test_quants_start_at_offset_2(self) -> None:
        data = _mixed_block()
        quantized = quantize_q8_0(data)
        assert len(quantized) == BLOCK_Q8_0_SIZE
        for j in range(QK8_0):
            assert isinstance(quantized[2 + j], int)

    def test_two_blocks_layout(self) -> None:
        data = _mixed_block() + _zero_block()
        quantized = quantize_q8_0(data)
        assert len(quantized) == 2 * BLOCK_Q8_0_SIZE
        first_block = quantized[:BLOCK_Q8_0_SIZE]
        second_block = quantized[BLOCK_Q8_0_SIZE:]
        d1_bits = first_block[0] | (first_block[1] << 8)
        d2_bits = second_block[0] | (second_block[1] << 8)
        assert _float16_to_float32(d1_bits) > 0
        assert _float16_to_float32(d2_bits) == 0.0
