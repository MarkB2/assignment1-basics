from pathlib import Path
from pprint import pprint
from typing import Any, cast
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from pprint import pprint

from cs336_basics.config import Config, VocabConfig, TokenizerConfig
from cs336_basics.runs import Runner
from cs336_basics.trainer import train_and_save

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"

from dataclasses import dataclass




# cs = ConfigStore.instance()
# cs.store(name="io_artifact", node=IOArtifact)
# # # cs.store(name="vocab_schema", group="vocab", node=VocabConfig)
# # cs.store(name="tokenizer_schema", group="tokenizer", node=TokenizerConfig)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def my_app(cfg : DictConfig) -> None:
    config = instantiate(cfg)
    pprint(config, indent=4)
    # config = cast(Configxx, OmegaConf.to_object(cfg))
    # pprint(config, indent=4)
    # print(type(config.stage.inputs[0]))
    print(type(config))
    runner = Runner(config)
    print(type(config.dir))
    print(config.stage.inputs.source)
    runner.run()
    #     train_and_save(vocab)
    #     hash = hash_file_read("", vocab.input_path)
    #     update_manifest(config.dir, **{"outputs": {vocab.input_path: hash}})
    # if stage == Stage.TOKENIZE:
    #     validate_input(config.parent, config.tokenizer.vocab_path)



if __name__ == "__main__":
    my_app()
