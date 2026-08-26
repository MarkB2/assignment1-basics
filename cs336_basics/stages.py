from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from cs336_basics.config import VocabConfig
from cs336_basics.trainer import train_and_save


@dataclass
class IOSpec:
    path: str
    needs_validate: bool = True
    hash: str | None = None


IOSpecList = list[IOSpec]


class Stage[C](ABC):
    def __init__(self, config: C) -> None:
        self.config = config

    @abstractmethod
    def get_input_specs(self) -> IOSpecList: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def get_output_specs(self) -> IOSpecList: ...


class VocabStage(Stage[VocabConfig]):
    @override
    def get_input_specs(self) -> IOSpecList:
        return [IOSpec(self.config.input_path, False)]

    @override
    def run(self):
        train_and_save(self.config)

    @override
    def get_output_specs(self) -> IOSpecList:
        return [IOSpec(self.config.vocab_path)]
