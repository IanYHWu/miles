"""sft_rollout renders a plain-string prompt as next-token prediction on the text."""

import pytest

from miles.rollout.sft_rollout import _plain_text_loss_mask


class _Tokenizer:
    eos_token = "<|im_end|>"
    unk_token_id = None
    vocab = {"<|im_end|>": 2, "<|endoftext|>": 3}

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [10 + len(word) for word in text.split()]}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)


def test_text_tokens_plus_the_eos_token_with_loss_after_the_first():
    ids, mask = _plain_text_loss_mask(_Tokenizer(), "a bb ccc", "<|endoftext|>")
    assert ids == [11, 12, 13, 3] and mask == [0, 1, 1, 1]


def test_the_tokenizer_eos_token_is_the_default():
    ids, _ = _plain_text_loss_mask(_Tokenizer(), "a", None)
    assert ids == [11, 2]


def test_an_unknown_eos_token_is_rejected():
    with pytest.raises(ValueError, match="not a token"):
        _plain_text_loss_mask(_Tokenizer(), "a", "<|nope|>")
