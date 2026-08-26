from dataclasses import replace
from pathlib import Path
from typing import Protocol

import pytest
from omegaconf import ValidationError

from cs336_basics.runs import Manifest, asdict, compute_hash, validate
from cs336_basics.stages import IOSpec, IOSpecList


class TmpFileFactory(Protocol):
    def __call__(self, name: str, text: str = "test") -> Path: ...


@pytest.fixture
def tmpfile_factory(tmp_path: Path) -> TmpFileFactory:
    def inner(name: str, text: str = "test") -> Path:
        tmp = tmp_path / name
        _ = tmp.write_text(text)
        return tmp

    return inner


def test_tmpfile_factory(tmpfile_factory: TmpFileFactory):
    t = tmpfile_factory("test.txt")
    assert t.read_text() == "test"


@pytest.fixture
def expected_list(tmpfile_factory: TmpFileFactory) -> IOSpecList:
    names = ["file1", "file2", "file3"]
    contexts = ["context1", "context2", "context3"]
    hashes = ["7b8341482ea5", "53c5077690eb", "439b89f79420"]
    specs = IOSpecList()
    for name, context, hash in zip(names, contexts, hashes):
        filepath = tmpfile_factory(name, context)
        specs.append(IOSpec(str(filepath), hash=hash))
    return specs


def test_asdict(expected_list: IOSpecList):
    assert isinstance(asdict(expected_list), dict)


@pytest.fixture
def fake_parent(expected_list: IOSpecList, tmp_path: Path):
    m = Manifest(dir=str(tmp_path), stage="", created_at="", hash_dict=asdict(expected_list))
    m.save()
    return m.dir


def test_compute_hash(expected_list: IOSpecList):
    new_specs = [replace(spec, hash=None) for spec in expected_list]
    assert compute_hash(new_specs) == expected_list


def test_validate(expected_list: IOSpecList, fake_parent: str):
    _ = validate(expected_list, fake_parent)


def test_validate_wrong_parent(expected_list: IOSpecList):
    with pytest.raises(FileNotFoundError):
        _ = validate(expected_list, "non_existent")


def test_validate_none_parent(expected_list: IOSpecList):
    with pytest.raises(ValidationError):
        _ = validate(expected_list, None)


def test_validate_throw_exception_on_file_change(expected_list: IOSpecList, fake_parent: str):
    _ = Path(expected_list[0].path).write_text("new text")
    with pytest.raises(ValidationError) as exinfo:
        _ = validate(expected_list, fake_parent)


def test_validate_on_file_change_not_needed(expected_list: IOSpecList, fake_parent: str):
    _ = Path(expected_list[0].path).write_text("new text")
    expected_list[0].needs_validate = False
    _ = validate(expected_list, fake_parent)
