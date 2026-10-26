# HydroSense dataset contract

CSV or .xlsx worksheet, one timestamp column and arbitrary numeric tag columns.
The user maps operating inputs and predicted targets separately for each asset.
Use offset-bearing timestamps or explicitly configure the source timezone.
Headers and timestamps must be unique. Numeric data remains in source engineering
units; PI must supply the same units. Known unhealthy history must be excluded
before training. A minimum of 250 complete/running rows per asset is enforced.

No real plant datasets are included. The optional deterministic synthetic demo
contains operating load, speed, cooling, ambient conditions, bearing temperature
and vibration, generator temperature, and transformer temperatures. The example
names do not constrain uploaded files. Synthetic replay introduces bearing and
transformer deviations to exercise the workflow, not to estimate real accuracy.

Uploads remain in service memory. Locally trained models and configuration are
stored under ignored artifacts/hydro/. See README.md for file-size and timing
limits.
