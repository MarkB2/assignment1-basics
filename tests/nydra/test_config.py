from dataclasses import dataclass
from pathlib import Path

import pytest
from hydra.compose import compose
from hydra.initialize import initialize
from hydra.utils import get_class, instantiate
from omegaconf import ValidationError

from cs336_basics.config import Config
from cs336_basics.runs import Manifest, Runner, file_hash, update_manifest, validate_inputs, write_manifest


@pytest.fixture
def config():
    with initialize(version_base="1.3", config_path="../../configs"):
        yield compose(config_name="config")


def test_config_target(config):
    assert get_class(config._target_) == Runner


def test_instantiation(config):
    assert isinstance(instantiate(config), Runner)


@pytest.fixture
def runner(config):
    return instantiate(config)


@pytest.fixture
def temp_file(tmp_path):
    path = tmp_path / "temp_file"
    path.write_text("-".join(["test" for _ in range(100)]))
    return path


def test_file_hash(temp_file):
    assert file_hash(str(temp_file)) == "c91531e6f0fe"
    with temp_file.open("a") as f:
        f.write("one more test")
    assert file_hash(str(temp_file)) != "c91531e6f0fe"


def test_validate_sources_false(temp_file):
    sources = {str(temp_file): "c91531e6f0fe"}
    inputs = {str(temp_file): False}
    input_hashes = validate_inputs(sources, inputs)
    assert input_hashes == {str(temp_file): None}


def test_validate_sources_true(temp_file):
    sources = {str(temp_file): "c91531e6f0fe"}
    inputs = {str(temp_file): True}
    input_hashes = validate_inputs(sources, inputs)
    assert input_hashes == {str(temp_file): "c91531e6f0fe"}


def test_validate_sources_non_exist(temp_file):
    sources = {str(temp_file): "c91531e6f0fe"}
    inputs = {"doesnt_exist": True}
    with pytest.raises(ValidationError):
        validate_inputs(sources, inputs)


def test_validate_sources_wrong_hash(temp_file):
    sources = {str(temp_file): "c91531e6f0ff"}
    inputs = {str(temp_file): True}
    with pytest.raises(ValidationError):
        validate_inputs(sources, inputs)


@pytest.fixture
def fake_manifest(tmp_path):
    return Manifest(
        dir=str(tmp_path),
        stage="fake manifest",
        created_at="",
        git_commit="",
        parent=None,
    )


def test_write_manifest(fake_config, temp_file):
    input_hashes = {str(temp_file): "c91531e6f0fe"}
    write_manifest(fake_config, input_hashes)
    assert Manifest.load(fake_config.current_path)


@dataclass
class FakeStage:
    name: str = "Fake Stage"


@pytest.fixture
def fake_config(tmp_path):
    return Config(current_path=str(tmp_path), stage=FakeStage())


def test_update_manifest(fake_manifest, temp_file):
    fake_manifest.save()
    update_manifest(fake_manifest.dir, [str(temp_file)])
    assert Manifest.load(fake_manifest.dir).outputs == {str(temp_file): "c91531e6f0fe"}
