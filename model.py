"""

main branch1

# I want it to be more heavily related to flow times delta t
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
        # output: (B_sorted, max_T_unpacked, hidden_size)

        correction = self.fc(output)        # (B_sorted, T, 1)
        correction = torch.tanh(correction) * 0.3   # bound to ±30%

        _, unsort_idx = sort_idx.sort()
        correction = correction[unsort_idx]

        # Naive cumulative volume is feature index 3
        T_out = correction.size(1)
        naive_vol = x_padded[:, :T_out, 3:4]   # (B, T, 1)

        volumes = naive_vol * (1.0 + correction)
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
        correction = self.fc(out.squeeze(0))       # (1, 1)
        correction = torch.tanh(correction).item() * 0.3

        naive_vol = x_row[3].item()                # feature 3 = cumsum(flow*dt)
        return naive_vol * (1.0 + correction)