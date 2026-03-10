

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    data_dir: str = "data"        
    eval_dir: str = "test_data"  
    model_save_path: str = "best_model.pt"
    results_dir: str = "results"

    target_volume: float = 3.0          
    val_split: float = 0.2       
    seed: int = 42

    # edit this -------------------------------------------------------------
    input_size: int = 3                 
    hidden_size: int = 64
    num_layers: int = 2
    fc_size: int = 32
    dropout: float = 0.2

    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"             # try to find best output bewtween "adam"  "adamw"  "sgd"

    def __post_init__(self):
        Path(self.results_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.eval_dir).mkdir(exist_ok=True)