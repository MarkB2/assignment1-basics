from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, override

from cs336_basics.config import VocabConfig
from cs336_basics.trainer import train_and_save


@dataclass
class IOSpec:
    path: str
    need_validate: bool = True
    hash: str | None = None


C = TypeVar("C")


class Stage(ABC, Generic[C]):
    def __init__(self, config: C) -> None:
        self.config = config

    @abstractmethod
    def get_input_specs(self) -> list[IOSpec]: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def get_output_specs(self) -> list[IOSpec]: ...


class VocabStage(Stage[VocabConfig]):
    @override
    def get_input_specs(self) -> list[IOSpec]:
        return [IOSpec(self.config.input_path, False)]

    @override
    def run(self):
        train_and_save(self.config)

    @override
    def get_output_specs(self) -> list[IOSpec]:
        return [IOSpec(self.config.input_path, False), IOSpec(self.config.vocab_path)]
