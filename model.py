

"""
<<<<<<< HEAD
Test branch
=======


forward()  → processes full sequence, returns all volumes
step()     → processes one row at a time
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpirometryLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2,
                 fc_size=32, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first =True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1),          
        )

        self._h = None
        self._c = None


    def forward(self, x_padded, lengths):

        sorted_len, sort_idx = lengths.sort(descending=True)
        x_sorted = x_padded[sort_idx]

        packed = pack_padded_sequence(x_sorted, sorted_len.cpu(),
                                      batch_first=True)
        packed_out, _ = self.lstm(packed)

        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        volumes = self.fc(output)           # batch, max_T, 1

        _, unsort_idx = sort_idx.sort()
        volumes = volumes[unsort_idx]
        return volumes

    def reset_state(self, device=None):
        self._h = None
        self._c = None

    @torch.no_grad()
    def step(self, x_row, device=None):
        if device is None:
            device = x_row.device

        x = x_row.unsqueeze(0).unsqueeze(0).to(device)

        if self._h is None:
            self._h = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
            self._c = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)

        out, (self._h, self._c) = self.lstm(x, (self._h, self._c))
        vol = self.fc(out.squeeze(0))       # (1, 1)
        return vol.item()