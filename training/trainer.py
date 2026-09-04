import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from training.ema import Ema


class Trainer:
    """BCE training loop with AMP, cosine schedule and EMA weights; validation
    runs on un-augmented data with the EMA weights, and the best epoch is
    selected by validation log loss (the competition metric)."""

    def __init__(self, model, device, epochs=40, lr=3e-4, weight_decay=1e-4,
                 ema_decay=0.999, label_smoothing=0.0, loss="bce", focal_gamma=2.0,
                 log_fn=print):
        self.loss_name = loss
        self.focal_gamma = focal_gamma
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.label_smoothing = label_smoothing
        self.log = log_fn

    def fit(self, train_loader, val_loader):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs * max(1, len(train_loader))
        )
        use_amp = self.device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if self.loss_name == "focal":
            def criterion(logits, y):
                ce = nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none")
                pt = torch.exp(-ce)
                return ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            criterion = nn.BCEWithLogitsLoss()
        ema = Ema(self.model, self.ema_decay)

        best = {"log_loss": float("inf"), "state": None, "epoch": -1}
        history = []
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for x, y in train_loader:
                x, y = self._to_device(x), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                if self.label_smoothing > 0:
                    y = y * (1 - 2 * self.label_smoothing) + self.label_smoothing
                with torch.autocast("cuda", enabled=use_amp):
                    loss = criterion(self.model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(self.model)
                losses.append(loss.item())

            ema.copy_to(self.model)
            val_probs, val_y = self.predict(val_loader)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_log_loss": float(log_loss(val_y, np.clip(val_probs, 1e-6, 1 - 1e-6))),
                "val_auroc": float(roc_auc_score(val_y, val_probs)),
                "lr": sched.get_last_lr()[0],
            }
            if row["val_log_loss"] < best["log_loss"]:
                best = {
                    "log_loss": row["val_log_loss"],
                    "state": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                    "epoch": epoch,
                }
            ema.restore(self.model)
            history.append(row)
            self.log(
                f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  "
                f"val_ll {row['val_log_loss']:.4f}  val_auc {row['val_auroc']:.4f}"
            )
        return best, history

    def _to_device(self, x):
        if isinstance(x, (list, tuple)):
            return [t.to(self.device, non_blocking=True) for t in x]
        return x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        probs, ys = [], []
        for x, y in loader:
            x = self._to_device(x)
            with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                logits = self.model(x)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            ys.append(y.numpy())
        return np.concatenate(probs), np.concatenate(ys)
