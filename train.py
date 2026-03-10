
"""
"""

import json
import torch
import torch.nn as nn
from config import Config
from dataset import get_dataloaders
from model import SpirometryLSTM


def build_optimizer(model, cfg):
    params = model.parameters()
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=cfg.learning_rate,
                               weight_decay=cfg.weight_decay, momentum=0.9)
    else:

        raise ValueError(f"Unknown optimizer: {name}")


def masked_mse_loss(preds, labels, lengths):
    batch_size, max_T = labels.shape
    mask = torch.arange(max_T, device=labels.device).unsqueeze(0) < lengths.unsqueeze(1)
    preds_flat  = preds.squeeze(-1)[mask]       
    labels_flat = labels[mask]                   
    return nn.functional.mse_loss(preds_flat, labels_flat)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x_pad, lengths, labels in loader:
        x_pad, lengths, labels = x_pad.to(device), lengths.to(device), labels.to(device)
        preds = model(x_pad, lengths)
        loss = masked_mse_loss(preds, labels, lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()
    total_loss = 0.0
    all_final_preds, all_final_labels = [], []
    n = 0
    for x_pad, lengths, labels in loader:
        x_pad, lengths, labels = x_pad.to(device), lengths.to(device), labels.to(device)
        preds = model(x_pad, lengths)
        loss = masked_mse_loss(preds, labels, lengths)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        # Collect final timestep prediction for each sample
        for i in range(labels.size(0)):
            t_last = int(lengths[i].item()) - 1
            all_final_preds.append(preds[i, t_last, 0].cpu().item())
            all_final_labels.append(labels[i, t_last].cpu().item())

    avg_loss = total_loss / max(n, 1)
    return avg_loss, all_final_preds, all_final_labels


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)

    model = SpirometryLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        fc_size=cfg.fc_size,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()  # kept for reference, masked version used internally
    optimizer = build_optimizer(model, cfg)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.model_save_path)
            marker = " ★ saved"

        print(f"[{epoch:3d}/{cfg.epochs}]  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

    # Save training history for evaluate.py
    with open(f"{cfg.results_dir}/history.json", "w") as f:
        json.dump(history, f)

    # Save final val predictions for scatter plot
    val_loss, val_preds, val_labels = evaluate(model, val_loader, device)
    with open(f"{cfg.results_dir}/val_predictions.json", "w") as f:
        json.dump({"preds": val_preds, "labels": val_labels}, f)

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Model saved to {cfg.model_save_path}")
    print(f"Graphs data saved to {cfg.results_dir}/")


if __name__ == "__main__":
    main()