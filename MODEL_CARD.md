# HydroSense model card

- Model: per-asset multi-output Extra Trees conditional regression.
- Purpose: expected-versus-actual monitoring of numeric PI tags at current operating conditions.
- Inputs and outputs: arbitrary user-selected tag names; targets cannot be inputs to the same model.
- Training: explicitly confirmed healthy rows only; optional running threshold.
- Split: chronological 60% fit, 20% calibration, 20% held-out evaluation.
- Threshold: per-tag 99th-percentile calibration absolute residual, with a small numerical floor.
- Evaluation: held-out MAE, asset flagged fraction and input range coverage; no failure-accuracy claim.
- Runtime: shared scoring path for imported CSV/Excel replay and live PI snapshots.
- Abstention: missing/bad/stale/misaligned signals, stopped asset, unseen input ranges.
- Alerts: configurable consecutive fresh deviations; no diagnosis or automatic plant control.
- Limitations: no temporal dynamics/lag features, no future forecasts, no automatic retraining,
  no labelled-failure validation, no probability interpretation or remaining-life prediction.
- Validity: real plant data and the actual PI deployment were unavailable during implementation.
  Synthetic tests establish software behaviour, not operational predictive performance.

Full setup, data and operational details are in README.md.
