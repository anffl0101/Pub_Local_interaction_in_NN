# A preference for local interaction between layers emerges during training

Code for the results in arXiv:2611.27258.

 Five-layer MLPs (width 256) learn modular addition, (a + b) mod 113, trained with AdamW or SGLD; 512 networks of each. One layer of a copy of a network is pushed, and the code measures how much of the push reaches the other layers. The clock is the training step: early = pushes up to step 15,000, late = pushes from step 60,000.

If you use this code, please cite: J. Lee, "A preference for local interaction between layers emerges during training", arXiv:2611.27258 (2026).

## Layout

- `src/` model, data, the settings of the two optimizers, training of many networks at once, helpers (folders, seeds, configs).
- `scripts/runs/` training and push runs, and the runs on saved states (line-up, one-step response, curvature).
- `scripts/figures/` the scripts that make the figures and tables of the paper, and the modules they use.
- `configs/default.json` task settings (p = 113, training fraction 0.4, split seed 0).
- `figures/` the figures of the paper.

The data are not included. The scripts read them from `data/<data set>/` and the saved network states from `checkpoints/` (or `$GROKKING_CKPT`).
