"""Token counting helper.

`cl100k_base` (GPT-3.5/4 encoding) is used as a generic cross-model
approximation. The actual count for non-OpenAI models will differ, but
the per-second metric stays comparable across models because the same
encoder is applied uniformly.

When the OpenAI-compatible API returns `usage.completion_tokens` we
prefer that — it reflects the model's real tokenizer. tiktoken is the
fallback path (and the only path for older JSON dumps that predate the
usage-passthrough change).
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text, disallowed_special=()))
