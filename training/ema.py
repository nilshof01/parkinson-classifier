import torch


class Ema:
    """Exponential moving average over the full state dict (params + buffers);
    validation and model selection run on the EMA weights (training.md)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        # decay warmup: without it, short runs (few steps/epoch) would leave the
        # EMA dominated by the random initialization
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(
            {k: v.to(dtype=b.dtype) for (k, v), b in zip(self.shadow.items(), self._backup.values())}
        )

    def restore(self, model):
        model.load_state_dict(self._backup)
        self._backup = None

    def state_dict(self):
        return dict(self.shadow)
