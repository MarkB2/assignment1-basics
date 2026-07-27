import pytest
import json
from pathlib import Path
import numpy as np

from ..common import FIXTURES_PATH
from cs336_basics.types import pack_pair, unpack_pair
from cs336_basics.tokens import Word
from cs336_basics.vocab import Vocab, string_vocab_to_bytes_vocab, bytes_vocab_to_vocab, StringVocab
from cs336_basics.tokenizer import Tokenizer

VOCAB_PATH = FIXTURES_PATH / 'vocab.json'
SPECIAL_TOKENS = ["<|endoftext|>", "<|endofphrase|>"]

def assert_valid(tokenizer: Tokenizer, special_tokens: list[str] | None):
    assert tokenizer
    for item in special_tokens or []:
        assert tokenizer.vocab.id_for(bytes(item, encoding='utf-8')) >= 0

def load_dict_vocab(path: Path | str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(path) as f:
        return string_vocab_to_bytes_vocab(StringVocab(*json.load(f)))

def word_helper(vocab: Vocab, text:str) -> Word:
    return Word(vocab.to_ids(text))

def bytes_helper(vocab: Vocab, ids: list[int]) -> list[bytes]:
    return [vocab.bytes_for(id) for id in ids]



@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(Vocab(), SPECIAL_TOKENS)

def test_create(tokenizer: Tokenizer):
    assert_valid(tokenizer, SPECIAL_TOKENS)

def test_from_file():
    tokenizer = Tokenizer.from_file(VOCAB_PATH, SPECIAL_TOKENS)
    assert_valid(tokenizer, SPECIAL_TOKENS)

def test_from_dicts():
    with open(VOCAB_PATH) as f:
        vocab, merges = load_dict_vocab(VOCAB_PATH)
        assert_valid(Tokenizer.from_dicts(vocab, merges, SPECIAL_TOKENS), SPECIAL_TOKENS)

@pytest.fixture
def file_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(VOCAB_PATH, SPECIAL_TOKENS)


def test_best_pair(file_tokenizer: Tokenizer) -> None:
    tokens = word_helper(file_tokenizer.vocab, "collaboration").tokens.tolist()
    pos, replacement = file_tokenizer.best_pair(tokens)
    assert pos == 8
    assert replacement == 267

def test_encode(file_tokenizer: Tokenizer) -> None:
    text = "collaboration"
    tokens = file_tokenizer.encode(text)
    assert bytes_helper(file_tokenizer.vocab, tokens) == [b'c', b'ol', b'l', b'a', b'b', b'or', b'ation']

def test_encode_iterable(file_tokenizer: Tokenizer) -> None:
    texts = ["collaboration", " revolution"]
    tokens = list(file_tokenizer.encode_iterable(texts))
    assert bytes_helper(file_tokenizer.vocab, tokens) == [
        b'c', b'ol', b'l', b'a', b'b', b'or', b'ation', b' re', b'v', b'ol', b'ut', b'ion'
    ]

def test_decode(file_tokenizer: Tokenizer) -> None:
    text = "collaboration and revolution"
    tokens = file_tokenizer.encode(text)
    assert file_tokenizer.decode(tokens) == text
