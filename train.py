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
    """Original data-only loss (kept for validation)."""
    batch_size, max_T = labels.shape
    mask = torch.arange(max_T, device=labels.device).unsqueeze(0) < lengths.unsqueeze(1)
    preds_flat  = preds.squeeze(-1)[mask]
    labels_flat = labels[mask]
    return nn.functional.mse_loss(preds_flat, labels_flat)


def pinn_loss(preds, labels, lengths, x_padded, cfg, epoch):
    """
    Combined PINN loss:
      L_data:     masked MSE against per-timestep labels
      L_physics:  dV/dt should match flow*dt
      L_mono:     penalize any predicted volume decrease
      L_boundary: V(0)~0 and V(T)~target_volume

    Physics terms ramp in over first `pinn_warmup_epochs` epochs.
    """
    B, max_T = labels.shape
    device = labels.device

    # ── Masks ─────────────────────────────────────────────
    mask_t = torch.arange(max_T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    # ── 1. Data loss ──────────────────────────────────────
    preds_sq = preds.squeeze(-1)                          # (B, max_T)
    L_data = nn.functional.mse_loss(preds_sq[mask_t], labels[mask_t])

    # ── Align pred/input lengths ──────────────────────────
    T_pred = preds_sq.size(1)
    x_clip = x_padded[:, :T_pred, :]

    # mask for consecutive valid pairs (t, t+1)
    mask_pair = mask_t[:, 1:T_pred] & mask_t[:, :T_pred - 1]   # (B, T_pred-1)

    # ── 2. Physics loss: dV_pred should match dV_label ─────
    # Labels are already scaled to 0→target_volume (litres),
    # so label diffs are the true per-step volume increments.
    # Raw flow*dt is in arbitrary sensor units ≠ litres, so we
    # compare against label increments instead.
    dV_pred  = preds_sq[:, 1:] - preds_sq[:, :-1]           # (B, T_pred-1)
    dV_label = labels[:, 1:T_pred] - labels[:, :T_pred - 1] # (B, T_pred-1)

    if mask_pair.any():
        L_physics = nn.functional.mse_loss(dV_pred[mask_pair], dV_label[mask_pair])
    else:
        L_physics = torch.tensor(0.0, device=device)

    # ── 3. Monotonicity ──────────────────────────────────
    drops = torch.clamp(preds_sq[:, :-1] - preds_sq[:, 1:], min=0.0)
    if mask_pair.any():
        L_mono = (drops[mask_pair] ** 2).mean()
    else:
        L_mono = torch.tensor(0.0, device=device)

    # ── 4. Boundary conditions ────────────────────────────
    L_start = preds_sq[:, 0].pow(2).mean()

    final_idx = (lengths - 1).long().clamp(min=0, max=T_pred - 1)
    final_preds = preds_sq[torch.arange(B, device=device), final_idx]
    L_end = (final_preds - cfg.target_volume).pow(2).mean()

    # ── Warm-up ramp ──────────────────────────────────────
    warmup = cfg.pinn_warmup_epochs
    ramp = min(1.0, epoch / warmup) if warmup > 0 else 1.0

    # ── Weighted sum ──────────────────────────────────────
    loss = (
        L_data
        + cfg.lambda_physics * ramp * L_physics
        + cfg.lambda_mono   * ramp * L_mono
        + cfg.lambda_end    * ramp * L_end
        + cfg.lambda_start  * ramp * L_start
    )

    components = {
        "data":    L_data.item(),
        "physics": L_physics.item(),
        "mono":    L_mono.item(),
        "start":   L_start.item(),
        "end":     L_end.item(),
        "ramp":    ramp,
    }
    return loss, components


def train_one_epoch(model, loader, optimizer, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    total_comp = {}
    n = 0
    for x_pad, lengths, labels in loader:
        x_pad   = x_pad.to(device)
        lengths = lengths.to(device)
        labels  = labels.to(device)

        preds = model(x_pad, lengths)
        loss, comp = pinn_loss(preds, labels, lengths, x_pad, cfg, epoch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        for k, v in comp.items():
            total_comp[k] = total_comp.get(k, 0.0) + v * bs

    avg_loss = total_loss / max(n, 1)
    avg_comp = {k: v / max(n, 1) for k, v in total_comp.items()}
    return avg_loss, avg_comp


@torch.no_grad()
def evaluate(model, loader, device):
    """Validation with data-only MSE for clean comparison."""
    model.eval()
    total_loss = 0.0
    all_final_preds, all_final_labels = [], []
    n = 0
    for x_pad, lengths, labels in loader:
        x_pad   = x_pad.to(device)
        lengths = lengths.to(device)
        labels  = labels.to(device)

        preds = model(x_pad, lengths)
        loss = masked_mse_loss(preds, labels, lengths)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

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
    print(f"PINN: physics={cfg.lambda_physics}, mono={cfg.lambda_mono}, "
          f"start={cfg.lambda_start}, end={cfg.lambda_end}, warmup={cfg.pinn_warmup_epochs}")

    train_loader, val_loader = get_dataloaders(cfg)

    model = SpirometryLSTM(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        fc_size=cfg.fc_size,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = build_optimizer(model, cfg)

    history = {"train_loss": [], "val_loss": [], "components": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, comp = train_one_epoch(
            model, train_loader, optimizer, device, cfg, epoch
        )
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["components"].append(comp)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.model_save_path)
            marker = " ★ saved"

        if epoch % 10 == 1 or epoch == cfg.epochs:
            print(f"[{epoch:3d}/{cfg.epochs}]  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"d={comp['data']:.5f} p={comp['physics']:.5f} "
                  f"m={comp['mono']:.6f} e={comp['end']:.5f} "
                  f"ramp={comp['ramp']:.2f}{marker}")
        else:
            print(f"[{epoch:3d}/{cfg.epochs}]  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}{marker}")

    with open(f"{cfg.results_dir}/history.json", "w") as f:
        json.dump(history, f)

    val_loss, val_preds, val_labels = evaluate(model, val_loader, device)
    with open(f"{cfg.results_dir}/val_predictions.json", "w") as f:
        json.dump({"preds": val_preds, "labels": val_labels}, f)

    print(f"\nDone. Best val loss: {best_val:.6f}")
    print(f"Model saved to {cfg.model_save_path}")


if __name__ == "__main__":
    main()