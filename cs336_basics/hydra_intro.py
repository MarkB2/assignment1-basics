from pathlib import Path
from pprint import pprint
from typing import cast
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from pprint import pprint

from cs336_basics.config import StageConfig, Config, VocabConfig, TokenizerConfig
from cs336_basics.runs import write_manifest, validate_input, hash_file_read, update_manifest
from cs336_basics.trainer import train_and_save

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="vocab_schema", group="vocab", node=VocabConfig)
cs.store(name="tokenizer_schema", group="tokenizer", node=TokenizerConfig)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def my_app(cfg : DictConfig) -> None:
    config = cast(Config, OmegaConf.to_object(cfg))
    pprint(config, indent=4)
    write_manifest(config)
    stage = config.stage
    if stage == StageConfig.VOCAB:
        vocab = cast(VocabConfig, config.vocab)
        train_and_save(vocab)
        hash = hash_file_read("", vocab.input_path)
        update_manifest(config.dir, **{"outputs": {vocab.input_path: hash}})
    if stage == StageConfig.TOKENIZE:
        validate_input(config.parent, config.tokenizer.vocab_path)



if __name__ == "__main__":
    my_app()
