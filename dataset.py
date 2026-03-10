


import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split


class SpirometryDataset(Dataset):

    def __init__(self, data_dir: str, target_volume: float = 3.0):
        self.samples: List[torch.Tensor] = []
        self.labels:  List[torch.Tensor] = []

        csv_files = sorted(Path(data_dir).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        for f in csv_files:
            df = pd.read_csv(f)

            delta_t = df["delta_t_ms"].values / 1000.0
            flow    = df["processed"].values

            # Strip 0
            mask = flow > 0.0
            flow    = flow[mask]
            delta_t = delta_t[mask]

            if len(flow) < 2:
                continue

            features = np.stack([flow, delta_t, flow * delta_t], axis=1)

            # Cumulative volume at each timestep
            cum_vol = np.cumsum(flow * delta_t)
            total_naive = cum_vol[-1]
            if total_naive <= 0:
                continue

            cum_labels = (cum_vol / total_naive) * target_volume       

            self.samples.append(torch.tensor(features, dtype=torch.float32))
            self.labels.append(torch.tensor(cum_labels, dtype=torch.float32))

        print(f"Loaded {len(self.samples)} recordings from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in sequences])

    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_lbl = pad_sequence(labels,    batch_first=True, padding_value=0.0)
    return padded_seq, lengths, padded_lbl


def get_dataloaders(cfg):
    dataset = SpirometryDataset(cfg.data_dir, cfg.target_volume)

    val_size   = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size

    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


