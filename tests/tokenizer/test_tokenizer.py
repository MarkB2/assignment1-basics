import pytest
import json
from pathlib import Path
from ..common import FIXTURES_PATH
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
