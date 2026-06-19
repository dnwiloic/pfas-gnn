"""PFAS / groundwater prediction — testable CPU modules (cf. CLAUDE.md §4).

Pipeline logic lives here (load, clean, targets, splits, features) so the SAME code
is mini-tested locally on CPU and run end-to-end on Colab GPU. Notebooks only
orchestrate (config, GPU, full run, save).

The shared "contract" frozen by the profiling / eval-methodology phase
(experiments/profilage/) lives in `config.py`:
  - target-leakage blocklist (96 cols),
  - EPA-2024 regulatory thresholds + detection guard (eval condition C1),
  - group key `gm_well_id` (C2) and spatial block CV (C3),
  - feature families with `gm_dataset_name` neutralized (C6).
"""

__all__ = ["config", "data", "targets", "splits", "features"]
