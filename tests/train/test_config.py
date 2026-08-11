from cs336_basics.train import ModelParms

from ..common import FIXTURES_PATH
from omegaconf import OmegaConf

model_cfg = FIXTURES_PATH / "model_cfg.yaml"

def test_omega_config():
    cfg = OmegaConf.load(model_cfg)
    assert cfg

def test_structured_config():
    schema = OmegaConf.structured(ModelParms)
    cfg = OmegaConf.load(model_cfg)
    config = OmegaConf.merge(cfg, schema)
    assert config
