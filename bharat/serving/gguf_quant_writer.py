from __future__ import annotations

import math
import struct
from collections.abc import Sequence

QK8_0 = 32
BLOCK_Q8_0_SIZE = 34


def _float32_to_float16(value: float) -> int:
    bits = struct.pack("<f", value)
    f32 = int.from_bytes(bits, "little")
    sign = (f32 >> 16) & 0x8000
    exp = (f32 >> 23) & 0xFF
    mant = f32 & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError("NaN and Inf cannot be converted to float16")
    if exp == 0:
        return sign
    new_exp = exp - 127 + 15
    if new_exp >= 31:
        return sign | 0x7C00
    if new_exp <= 0:
        return sign
    new_mant = mant >> 13
    round_bit = (mant >> 12) & 1
    new_mant += round_bit
    if new_mant & 0x400:
        new_mant = 0
        new_exp += 1
        if new_exp >= 31:
            return sign | 0x7C00
    return sign | (new_exp << 10) | new_mant


def _float16_to_float32(bits: int) -> float:
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        exp = 0
        while mant >> 10 == 0:
            mant <<= 1
            exp -= 1
        mant &= 0x3FF
        exp += 1
    if exp == 31:
        raise ValueError("NaN and Inf cannot be represented in float32 from float16")
    f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", f32))[0]  # type: ignore[no-any-return]


def _round_half_away_from_zero(x: float) -> int:
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def quantize_q8_0(data: Sequence[float]) -> bytes:
    if not data:
        raise ValueError("data must not be empty")
    if len(data) % QK8_0 != 0:
        raise ValueError(f"data length ({len(data)}) must be a multiple of {QK8_0}")
    blocks: list[bytes] = []
    for block_start in range(0, len(data), QK8_0):
        amax = 0.0
        for j in range(QK8_0):
            v = abs(data[block_start + j])
            if v > amax:
                amax = v
            if math.isnan(v) or math.isinf(v):
                raise ValueError("data contains NaN or Inf")
        d = amax / 127.0
        d_bits = _float32_to_float16(d)
        block = bytearray(2 + QK8_0)
        block[0] = d_bits & 0xFF
        block[1] = (d_bits >> 8) & 0xFF
        if d > 0:
            id_ = 1.0 / d
            for j in range(QK8_0):
                q = _round_half_away_from_zero(data[block_start + j] * id_)
                q = max(-128, min(127, q))
                block[2 + j] = q & 0xFF
        blocks.append(bytes(block))
    return b"".join(blocks)


def dequantize_q8_0(data: bytes, element_count: int) -> list[float]:
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    if element_count % QK8_0 != 0:
        raise ValueError(f"element_count ({element_count}) must be a multiple of {QK8_0}")
    expected_bytes = (element_count // QK8_0) * BLOCK_Q8_0_SIZE
    if len(data) != expected_bytes:
        raise ValueError(
            f"data length ({len(data)}) does not match expected length "
            f"({expected_bytes}) for {element_count} elements"
        )
    result: list[float] = []
    for block_start in range(0, len(data), BLOCK_Q8_0_SIZE):
        d_bits = data[block_start] | (data[block_start + 1] << 8)
        d = _float16_to_float32(d_bits)
        for j in range(QK8_0):
            q = data[block_start + 2 + j]
            q_signed = q - 256 if q >= 128 else q
            result.append(q_signed * d)
    return result
