# Quadrotor Dynamics Calibration

Code for **From Forecast Accuracy to Control Utility: Evaluating Quadrotor Dynamics Calibration at the Decision Interface**.

- [Supplementary material](supplementary/supplementary.pdf)
- [Video](video/demo.mp4)

## Setup

Use Python 3.11. Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Plot the reported results

```bash
python scripts/plot_results.py --output figures
```

This creates Figures 2–5 in PDF, PNG and SVG, plus the Table I LaTeX fragment, from the supplied numerical summaries in `data/results/`. Choose a new output directory. This command reproduces the reported displays; it does not rerun simulations or estimate confidence intervals.

## Experiment code

| Component | Entry points |
|---|---|
| Vehicle simulation, control, wind fields and context models | `src/winddyn/`, `configs/` |
| Environment and training data generation | `scripts/generate_scenes.py`, `scripts/run_cfd_batch.py`, `scripts/collect_rollouts.py` |
| Context-encoder training | `scripts/revision_v2.py` |
| Calibration | `scripts/icra_confirm_frozen.py`, `scripts/icra_confirm_adapt.py` |
| Identical-state action evaluation | `scripts/icra_campaign_snapshot.py`, `scripts/icra_snapshot_branches.py` |
| Shared forecast and action-response analysis | `scripts/icra_action_decomposition.py`, `scripts/icra_cost_difference.py` |
| Feedback continuation | `scripts/icra_action_continuation.py` |
| Closed-loop evaluation | `scripts/icra_paired_mpc.py`, `scripts/icra_selection_run.py` |
| Nonlinear readout comparison | `scripts/icra_nonlinear_readout_20260914.py`, `scripts/icra_nonlinear_readout_summary_20260914.py` |

The fitted encoder and readout inputs are under `runs/`, preserving the relative paths used by the experimental code. Scene descriptions are under `data/manifests/`. The full training datasets, generated wind-field arrays and historical simulation-output archive are not bundled. Full experiment runs require the corresponding data and original run configuration/protocol inputs; the supplementary material describes the experimental settings.

CPU execution is sufficient for plotting and loading the supplied predictors. Collection and training scripts specify their own device options. Run experiments in a separate working copy so their generated outputs remain separate from the supplied fitted parameters.
