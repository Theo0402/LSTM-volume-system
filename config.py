from dataclasses import dataclass, field
from pathlib import Path

"""
pip install -r requirements.txt
"""


@dataclass
class Config:
    data_dir: str = "data"        
    eval_dir: str = "test_data"  
    model_save_path: str = "best_model.pt"
    results_dir: str = "results"

    target_volume: float = 3.0          
    val_split: float = 0.2       
    seed: int = 42

    input_size: int = 4                 
    hidden_size: int = 64
    num_layers: int = 2
    fc_size: int = 32
    dropout: float = 0.2

    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"             # try to find best output bewtween adam  adamw  sgd

    # PINN loss weights
    lambda_physics: float = 0.1         # dV/dt ≈ flow*dt
    lambda_mono: float = 1.0            # penalize volume decreases
    lambda_start: float = 0.3           # V(0) ≈ 0
    lambda_end: float = 0.5             # V(T) ≈ target_volume
    pinn_warmup_epochs: int = 30        # ramp physics terms over this many epochs

    def __post_init__(self):
        Path(self.results_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.eval_dir).mkdir(exist_ok=True)