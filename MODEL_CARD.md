# HydroSense model card

- Purpose: predict every numeric tag of a hydro unit and expose deviations for engineer review.
- Models: multi-target Temporal Fusion Transformer (10/50/90% forecast quantiles), LSTM sequence autoencoder and Isolation Forest.
- Context: all tags over the previous configured lookback; known time-of-day features. Future tag values are never supplied as decoder inputs.
- Outputs: all tags over the configured horizon, in original engineering units.
- Training: explicitly confirmed healthy history; optional running gate. Internal aliases support arbitrary tag names.
- Split: chronological 60% fit, 20% calibration, 20% held-out evaluation; fit-only scaling and weights.
- Alerts: per-tag 99th-percentile absolute calibration residual; current actual compared to a forecast issued earlier. Independent fresh-observation persistence counters.
- Diagnostics: unit-level IF/LSTM scores normalized by healthy calibration thresholds; not failure probabilities.
- Evaluation: held-out full-horizon MAE per tag, last-value baseline MAE, one-step unit flagged fraction over sampled test origins.
- Missing/invalid data: affected unit abstains and warms up again; complete temporal windows are required.
- Runtime: shared forecast code for CSV/Excel replay and live PI snapshots. CPU inference, optional CUDA training through the notebook or configuration.
- Limitations: changing conditions and persistent faults can influence forecasts through recent history; no automatic retraining, fault classification or remaining-life estimates. Quantiles are not coverage guarantees. Thresholds need pilot validation.
- Validation: real CPU training and synthetic software tests; no plant data, live PI deployment or GPU execution was available for verification.

See README.md for setup and POC_PLAN.md for the SmartSignal comparison protocol.
