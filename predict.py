"""
python predict.py data/spirometry_2026-02-27T17-18-01-317515.csv             
"""

import sys
import numpy as np
import pandas as pd
import torch
from config import Config
from model import SpirometryLSTM



def load_model(cfg, device):
    model = SpirometryLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, 
        fc_size=cfg.fc_size,
        dropout=0.0,           # no dropout at inference
    ).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()
    return model


def realtime_from_csv(csv_path: str):
    """Simulate real-time: feed one row at a time using stateful step()."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    df = pd.read_csv(csv_path)
    delta_t = df["delta_t_ms"].values / 1000.0
    flow    = df["processed_real_units"].values

    print(f"{'Step':>5}  {'Flow':>10}  {'dt(s)':>8}  {'Pred Vol (L)':>12}")
    print("-" * 42)

    model.reset_state(device)
    vol = 0.0
    running_sum = 0.0

    for t in range(len(flow)):
        if flow[t] == 0.0:
            pass
        else:
            running_sum += flow[t] * delta_t[t]
            row = torch.tensor(
                [flow[t], delta_t[t], flow[t] * delta_t[t], running_sum],
                dtype=torch.float32,
            ).to(device)
            vol = model.step(row, device)

        print(f"{t+1:5d}  {flow[t]:10.4f}  {delta_t[t]:8.4f}  {vol:12.4f}")

    print(f"\nFinal predicted volume: {vol:.4f} L")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        realtime_from_csv(sys.argv[1])
    else:
        print("Usage: python predict.py <path/to/recording.csv>")