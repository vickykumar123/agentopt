"""The hashing recipes from schemas/hashing.md, pinned by known vectors.

The digests asserted here are cross-language contract values: the TypeScript
implementation must reproduce them exactly.
"""

from __future__ import annotations

from agentopt.hashing import content_hash, params_content_hash, prompt_hash

CLASSIFIER_MESSAGES = [
    {"role": "system", "content": "You classify fan messages into intents."},
    {"role": "user", "content": "money back pls??"},
]
CLASSIFIER_PROMPT_HASH = "7f1c695ef0cb1cee56ea36870adb656ac5cdb4e5b3721ab5ef829b56617a8ba3"


def test_prompt_hash_known_vector() -> None:
    assert prompt_hash(CLASSIFIER_MESSAGES) == CLASSIFIER_PROMPT_HASH


def test_prompt_hash_is_order_sensitive() -> None:
    assert prompt_hash(list(reversed(CLASSIFIER_MESSAGES))) != CLASSIFIER_PROMPT_HASH


def test_prompt_hash_non_text_content_uses_placeholder() -> None:
    with_image = [{"role": "user", "content": [{"type": "image", "data": "..."}]}]
    with_placeholder = [{"role": "user", "content": "[non-text]"}]
    assert prompt_hash(with_image) == prompt_hash(with_placeholder)


def test_content_hash_no_normalization() -> None:
    assert content_hash("You classify.") != content_hash("You classify. ")


def test_params_content_hash_ignores_key_order() -> None:
    a = {"temperature": 0.3, "max_output_tokens": 512}
    b = {"max_output_tokens": 512, "temperature": 0.3}
    assert params_content_hash(a) == params_content_hash(b)
