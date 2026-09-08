"""Shared deterministic text-window behavior for training and inference."""

from __future__ import annotations

DEFAULT_WINDOW_CHARS = 1800
DEFAULT_WINDOW_STRIDE = 900


def validate_window_parameters(window_chars: int, stride_chars: int) -> None:
    if window_chars < 1:
        raise ValueError("window_chars must be positive")
    if not 1 <= stride_chars <= window_chars:
        raise ValueError("stride_chars must be between 1 and window_chars")


def sliding_bounds(
    text_length: int,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    stride_chars: int = DEFAULT_WINDOW_STRIDE,
) -> list[tuple[int, int]]:
    validate_window_parameters(window_chars, stride_chars)
    if text_length <= 0:
        return []
    if text_length <= window_chars:
        return [(0, text_length)]
    bounds = []
    start = 0
    while start + window_chars < text_length:
        bounds.append((start, start + window_chars))
        start += stride_chars
    final = (text_length - window_chars, text_length)
    if not bounds or bounds[-1] != final:
        bounds.append(final)
    return bounds


def answer_centered_bounds(
    text_length: int,
    answer_start: int,
    answer_end: int,
    window_chars: int = DEFAULT_WINDOW_CHARS,
) -> tuple[int, int]:
    if not 0 <= answer_start < answer_end <= text_length:
        raise ValueError("answer span is outside the source text")
    if answer_end - answer_start > window_chars:
        raise ValueError("answer span is larger than the configured window")
    context_chars = window_chars - (answer_end - answer_start)
    start = max(0, answer_start - context_chars // 2)
    end = min(text_length, start + window_chars)
    start = max(0, end - window_chars)
    return start, end


def sliding_text_windows(
    text: str,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    stride_chars: int = DEFAULT_WINDOW_STRIDE,
) -> list[str]:
    return [text[start:end] for start, end in sliding_bounds(len(text), window_chars, stride_chars)]
