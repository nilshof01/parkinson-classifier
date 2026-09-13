Drop your trained checkpoint files here (state dicts saved with torch.save).

All .pt files found in this folder will be loaded with the architecture
defined in config.py and their predictions averaged.

Example — copy fold checkpoints from a run:
  cp output/runs/cnn3d_m4_baseline_s0/fold*/best_ema.pt submission/models/

Or copy a single checkpoint:
  cp output/runs/cnn3d_m4_baseline_s0/fold0/best_ema.pt submission/models/fold0.pt

The architecture settings in config.py (MODEL_POOL, MODEL_WIDTH_MULT, etc.)
must match what the checkpoints were trained with.
