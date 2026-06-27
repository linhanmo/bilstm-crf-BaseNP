from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_DATA_DIR = Path("ICWB2-BIOES") / "msr"
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass
class BaseTrainConfig:
    data_dir: str
    output_root: str
    model_name: str
    run_name: Optional[str] = None
    remove_o: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HMMTrainConfig(BaseTrainConfig):
    @classmethod
    def default(cls) -> "HMMTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name="hmm",
        )


@dataclass
class CRFTrainConfig(BaseTrainConfig):
    algorithm: str = "lbfgs"
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 100
    all_possible_transitions: bool = False

    @classmethod
    def default(cls) -> "CRFTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name="crf",
        )


@dataclass
class LSTMTrainConfig(BaseTrainConfig):
    batch_size: int = 64
    lr: float = 0.001
    epoches: int = 30
    print_step: int = 5
    emb_size: int = 128
    hidden_size: int = 128

    @classmethod
    def default(cls, model_name: str) -> "LSTMTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name=model_name,
        )
