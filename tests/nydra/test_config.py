from dataclasses import dataclass

from hydra.compose import compose
from hydra.initialize import initialize
from hydra.utils import get_class, instantiate
import pytest
from cs336_basics.config import Config
from cs336_basics.runs import Manifest, Runner, write_manifest

@pytest.fixture
def config():
    with initialize(version_base="1.3", config_path="../../configs"):
        yield compose(config_name="config")


def test_config_target(config):
    assert get_class(config._target_) == Config

def test_instantiation(config):
    assert isinstance(instantiate(config), Config)

@pytest.fixture
def runner(config):
    return Runner(instantiate(config))

@pytest.fixture
def fake_manifest(tmp_path):
    return Manifest(
        dir = str(tmp_path),
        stage = "fake manifest",
        created_at = "",
        git_commit = "",
        parent = None,
        inputs = {"my_path": "12"},
    )

@dataclass
class FakeStage:
    name: str = "Fake Stage"
    
@pytest.fixture
def fake_config(tmp_path):
    return Config(
        current_path = str(tmp_path),
        stage = FakeStage()
    )

def test_write_manifest(fake_config):
    write_manifest(fake_config)
    assert Manifest.load(fake_config.current_path)


def test_update_manifest(fake_manifest):
    fake_manifest.save()
    assert fake_manifest == Manifest.load(fake_manifest.dir)