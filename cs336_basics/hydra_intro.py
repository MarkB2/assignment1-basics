from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from cs336_basics.config import Config, VocabConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"

cs = ConfigStore.instance()
cs.store(name="main", node=Config)
cs.store(name="base", group="vocab", node=VocabConfig)


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base="1.3")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.job_type, cfg.job_name)

if __name__ == "__main__":
    my_app()
