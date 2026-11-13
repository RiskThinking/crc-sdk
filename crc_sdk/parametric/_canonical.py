"""Canonical JSON and content hashes, in crc-framework's spelling.

The rules are crc-framework's canonical profile (`crc-canonical-json/1`), the
one behind `ImpactExpression.definition_hash`, so a document hashes the same
in Python and in Rust:

- object keys sorted by their UTF-8 bytes, no whitespace;
- strings as UTF-8 without escaping non-ASCII; only the quote, the backslash
  and control characters below U+0020 are escaped;
- every number is a binary64 float, written as its shortest round-trip
  digits. Zero and negative zero are `0.0`. A number with a decimal exponent
  from -5 to 15 is written in positional notation with at least one digit
  after the point (`0.00001`, `1000000000000000.0`); any other in exponent
  notation without padding (`1e-6`, `1e+16`, `1.5e-7`);
- NaN and infinity are rejected.

The content hash is `"sha256:"` followed by the hex SHA-256 of the canonical
UTF-8 bytes.

A long sequence of plain numbers, or of strings, is spelled in one vectorised
step (`format_numbers`) rather than item by item, which makes hashing a large
outcome sample several times faster. Both paths give the same text.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["canonical_json", "content_hash", "format_number", "format_numbers"]

_MIN_POSITIONAL_EXPONENT = -5
_MAX_POSITIONAL_EXPONENT = 15

# Sequences at least this long take the vectorised path; shorter ones are
# cheaper item by item. The text is the same either way.
_BULK_LENGTH = 32

# `float.__repr__` writes the shortest round-trip digits in this layout too,
# except for decimal exponents from -5 to -9: it pads them (`1e-06`), and it
# writes -5 in exponent notation (`1.5e-05`, here `0.000015`). Every number
# repr writes with exponent -5 lies in this band of magnitudes.
_PADDED_EXPONENTS = tuple((f"e-0{digit}", f"e-{digit}") for digit in "6789")
_EXPONENT_MINUS_FIVE_BAND = (9e-6, 1.1e-4)


def _reject_non_finite(number: float) -> NoReturn:
    raise ValueError(f"canonical JSON has no spelling for {number!r}")


def format_number(value: float) -> str:
    """Return the canonical spelling of one finite number.

    Raises `ValueError` for NaN and infinity.
    """
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{value!r} is outside the binary64 range") from error
    if not math.isfinite(number):
        _reject_non_finite(number)
    if number == 0.0:
        return "0.0"
    sign = "-" if number < 0 else ""
    # repr gives the shortest digits that round-trip; only the layout differs.
    mantissa, _, exponent_text = repr(abs(number)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    point = len(whole) + (int(exponent_text) if exponent_text else 0)
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    exponent = point - 1
    if _MIN_POSITIONAL_EXPONENT <= exponent <= _MAX_POSITIONAL_EXPONENT:
        if point <= 0:
            return f"{sign}0.{'0' * -point}{digits}"
        if point >= len(digits):
            return f"{sign}{digits}{'0' * (point - len(digits))}.0"
        return f"{sign}{digits[:point]}.{digits[point:]}"
    head = digits[0] + (f".{digits[1:]}" if len(digits) > 1 else "")
    return f"{sign}{head}e{'-' if exponent < 0 else '+'}{abs(exponent)}"


def format_numbers(values: ArrayLike) -> list[str]:
    """Return the canonical spelling of each number of a 1-D array.

    The vectorised `format_number`: item by item the same text, for every
    binary64 value. Integers are read as binary64 floats, as `format_number`
    reads them. Raises `ValueError` for NaN, infinity, a value outside the
    binary64 range or an array that is not 1-D.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except OverflowError as error:
        raise ValueError("a value is outside the binary64 range") from error
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D array of numbers, got shape {array.shape}")
    if array.size == 0:
        return []
    return _spell_numbers(array).split(",")


def _spell_numbers(array: NDArray[np.float64]) -> str:
    """Return the canonical spellings of a non-empty 1-D array, comma-joined."""
    finite = np.isfinite(array)
    if not finite.all():
        _reject_non_finite(float(array[~finite][0]))
    # x + 0.0 is x for every x but negative zero, which becomes zero.
    array = array + 0.0
    if array.min() == array.max():
        return ",".join([format_number(float(array[0]))] * array.shape[0])
    # json writes each float as float.__repr__.
    text = json.dumps(array.tolist(), separators=(",", ":"))[1:-1]
    if "e-0" not in text:
        return text
    for padded, unpadded in _PADDED_EXPONENTS:
        text = text.replace(padded, unpadded)
    low, high = _EXPONENT_MINUS_FIVE_BAND
    magnitude = np.abs(array)
    band = np.flatnonzero((magnitude > low) & (magnitude < high))
    if band.size:
        tokens = text.split(",")
        for position in band.tolist():
            tokens[position] = format_number(float(array[position]))
        text = ",".join(tokens)
    return text


def _encode_bulk(value: Sequence[Any], parts: list[str]) -> bool:
    """Encode a sequence of plain numbers, or of strings, in one step.

    Returns False, having written nothing, for any other sequence, and for
    numbers the profile rejects, so the item-by-item path raises its error.
    """
    kinds = set(map(type, value))
    if kinds <= {float, int}:
        try:
            array = np.array(value, dtype=np.float64)
        except OverflowError:
            return False
        if not np.isfinite(array).all():
            return False
        parts.append(f"[{_spell_numbers(array)}]")
        return True
    if kinds == {str}:
        spelled = {item: _string(item) for item in set(value)}
        parts.append(f"[{','.join(map(spelled.__getitem__, value))}]")
        return True
    return False


def _string(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("canonical JSON strings must be valid Unicode") from error
    return json.dumps(value, ensure_ascii=False)


def _encode(value: Any, parts: list[str]) -> None:
    if value is None:
        parts.append("null")
    elif value is True:
        parts.append("true")
    elif value is False:
        parts.append("false")
    elif isinstance(value, str):
        parts.append(_string(value))
    elif isinstance(value, (int, float)):
        parts.append(format_number(value))
    elif isinstance(value, Mapping):
        keys: list[str] = []
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON keys must be strings, got {key!r}")
            _string(key)
            keys.append(key)
        parts.append("{")
        for position, key in enumerate(
            sorted(keys, key=lambda item: item.encode("utf-8"))
        ):
            if position:
                parts.append(",")
            parts.append(_string(key))
            parts.append(":")
            _encode(value[key], parts)
        parts.append("}")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) >= _BULK_LENGTH and _encode_bulk(value, parts):
            return
        parts.append("[")
        for position, item in enumerate(value):
            if position:
                parts.append(",")
            _encode(item, parts)
        parts.append("]")
    else:
        raise TypeError(f"canonical JSON cannot encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text of a document of plain JSON types.

    Mappings with string keys, sequences, strings, numbers, booleans and
    `None` are accepted. Integers are written as floats, as the profile
    reads every number as binary64. Raises `ValueError` for non-finite
    numbers and invalid Unicode, and `TypeError` for other types.
    """
    parts: list[str] = []
    _encode(value, parts)
    return "".join(parts)


def content_hash(value: Any) -> str:
    """Return `"sha256:<hex>"` of the canonical UTF-8 bytes of `value`."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
