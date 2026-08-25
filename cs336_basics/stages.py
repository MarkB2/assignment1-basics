from abc import ABC, abstractmethod
from typing import Generic, TypeVar, override
from dataclasses import dataclass

from cs336_basics.config import VocabConfig
from cs336_basics.trainer import train_and_save

@dataclass
class InputSpec:
    path: str
    validate: bool = True

@dataclass
class HashSpec:
    path: str
    hash: str | None = None
    
C = TypeVar("C")


class Stage(ABC, Generic[C]):
    def __init__(self, cfg: C) -> None:
        self.cfg = cfg
        
    @abstractmethod
    def get_input_specs(self) -> list[InputSpec]: ...

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def record_outputs(self) -> list[str]: ...

class VocabStage(Stage[VocabConfig]):

    @override
    def get_input_specs(self) -> list[InputSpec]:
        return [InputSpec(self.cfg.input_path, False)]

        
    @override
    def run(self):
        train_and_save(self.cfg)
        
    @override
    def record_outputs(self) -> list[str]:
        return [self.cfg.input_path, self.cfg.vocab_path]
