# HydroSense

Predictive maintenance pilot for hydro generators, bearings, transformers and
other unit assets. Import CSV or Excel history with arbitrary PI tag names.
**Every numeric tag is forecast simultaneously**, including operating variables
such as load and speed. The unit dashboard shows every tag's actual values,
previously issued predictions, future forecasts and independent deviations.

The modelling stack follows the AirSense approach: a Temporal Fusion Transformer
(TFT) forecasts all tags; an LSTM autoencoder and Isolation Forest add unit-level
anomaly diagnostics. Models learn from your historical data. In-app notifications
identify individual deviating tags while predictions continue for all tags.

[Download the public repository](https://github.com/bunjopolo/hydrosense/archive/refs/heads/main.zip)
· [Train in Google Colab](https://colab.research.google.com/github/bunjopolo/hydrosense/blob/main/notebooks/hydrosense_training.ipynb)

Upgrading from the earlier selected-target version: update the code, rerun
`python -m pip install -e .`, and retrain. Old regression artifacts cannot provide
all-tag temporal forecasts; the app retains them but asks for a new model.

## Start locally

Python 3.11+ and Node 22.13+ are required. From this directory:

```bash
python3 -m venv .venv                  # only if .venv does not already exist
source .venv/bin/activate
python -m pip install -r requirements-hydro.txt
python -m pip install --no-deps -e .
cd webapp
npm ci
cd ..
./scripts/run_dashboard.sh
```

Open http://127.0.0.1:5173. The launcher runs the local Python API on port 8765
and the dashboard on port 5173. Stop with Ctrl+C. Both services must be running;
the Python backend runs the models and stores the artifacts. This is a local
application; publishing only the web UI would not deploy the ML service.

For a Windows work computer, install Python 3.12, Node.js 22.13+ and Git, then
run the following in PowerShell from the cloned repository (no activation is
needed):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .
Push-Location webapp
npm ci
Pop-Location
$env:PYTHONPATH="src"
.venv\Scripts\python -m hydrosense.service
```

Leave that terminal open. In a second PowerShell terminal from the repository:

```powershell
cd webapp
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open http://localhost:5173. If your PI deployment uses Windows integrated
authentication, install its optional adapter with
`.venv\Scripts\python -m pip install requests-negotiate-sspi`, then choose Windows
integrated authentication in the app. The Python process must run as an account
with PI read access. The work computer must be able to reach the PI Web API.

The GitHub repository contains only HydroSense. Local copies of the former
AirSense implementation, plant files, trained models, generated reports and
connection settings are excluded from version control.

## Workflow

1. **Try the synthetic demo** on Monitor. It trains the actual neural pipeline
   on a ten-tag hydro unit, with deviations in a separate replay file. This is
   a software demonstration, not evidence of plant accuracy. It replaces the
   active configuration/model; previous model files remain on disk.
2. **Data & training:** choose the timestamp column, source timezone for naive
   timestamps, optional Excel sheet and unit name. Inspect a `.csv` or `.xlsx`
   file. **Use all tags for training** creates the unit and its complete tag list.
   There is no input/target split. Browser uploads are limited to 45 MB; use the
   notebook or CLI for larger files, subject to available memory.
3. Confirm the file contains reviewed healthy operation and train. Review each
   tag's held-out MAE against the last-value forecast baseline. A larger error
   than the baseline is a reason to investigate before trusting that model.
4. Load a separate replay file. All graphs share the same time window. Advance,
   play or seek the replay; forecasts start after the configured history window
   fills. Each actual is compared with a forecast issued before it arrived.
5. Download **Export pilot predictions** for timestamped per-tag predictions, residuals,
   thresholds and notification states. See [the pilot plan](POC_PLAN.md) for
   comparison with SmartSignal and engineer-reviewed events.
6. Enter PI settings in **PI connection**, test access, then start monitoring.
   Server details stay local. Synthetic demo models cannot be used against PI.

Use **Units & tags** to group columns into several units or adjust history,
forecast horizon, epochs and persistence. Every listed tag is both historical
model context and a forecast output. After editing configuration, reload the
file with that configuration and retrain. Changes invalidate the active model.

Import buffers, chart history and the notification list are in memory. Models,
configuration and durable prediction CSV logs are in ignored `artifacts/hydro/`.
The dashboard keeps 360 observations and the most recent 200 notifications;
logs retain the complete run. Notifications are in-app only.

## File contract

Wide format: one timestamp column and one column per tag. For example, these
names are illustrative; there are no required domain-specific names:

```csv
Time,Unit A MW,Bearing/Temperature,Ambient [C]
2026-01-01T00:00:00Z,72.1,56.2,12.4
2026-01-01T00:00:30Z,73.4,56.6,12.5
```

Excel uses the same layout, with headers in row 1. The selected sheet's cached
cell values are read; Excel formulas must have been recalculated and saved in
Excel first. Numeric strings are accepted. Non-numeric, blank, NaN or infinite
required readings are excluded during training and marked as quality issues
when scoring. String metadata columns can remain in the file, but should not be
selected as numeric model tags. Duplicate headers and timestamps are rejected.
Rows are sorted chronologically. Duplicate/conflicting records must be resolved
before import. No interpolation or future-value filling is performed.

Use ISO 8601 timestamps with offsets, or set `source_timezone` explicitly for
an entirely naive timestamp column (including Excel datetime cells). Mixed
aware/naive columns and ambiguous daylight-saving timestamps are rejected.
Choose a source timezone such as `America/Vancouver` rather than accepting UTC
if the exported values represent local plant time. Values are normalized to UTC.

Training needs at least **250 rows**, enough complete regularly sampled windows
in each chronological split, and at least one numeric tag. This is an execution
minimum, not evidence of operational quality. Use substantially more healthy
history spanning the intended operating envelope. Sampling cadence and engineering
units must match between training, replay and PI. Windows crossing missing values,
stopped periods or cadence gaps are excluded. No application tag-count cap is
imposed; memory and compute needs grow with the number of tags.

## Modelling and alerts

Each unit has one multi-target TFT. It predicts the next `horizon` samples from
`lookback` past samples, with 10th, 50th and 90th quantiles. External tag names
are mapped to internal aliases and preserved exactly in outputs. The earliest
60% of history fits models and scaling; the next 20% calibrates thresholds;
the latest 20% evaluates performance. Later splits may use earlier observations
as causal history, but their outcomes do not fit weights or scaling. Evaluation
uses non-overlapping forecast horizons. The report includes per-tag MAE over
the full horizon and a last-observation forecast baseline.

A separate 99th-percentile absolute calibration residual defines each tag's
alert tolerance at each forecast step, with a small numerical floor. A current
reading is compared with the most recently issued one-step forecast for its
exact timestamp. `abs(actual - predicted) / tolerance >= 1` marks a deviation.
Three consecutive fresh deviations produce a persistent alert by default.
Each tag has an independent counter; normal readings reset only that counter.
Quantile shading describes the forecast distribution; it is **not** the
calibrated notification threshold or a guarantee of coverage.

The Isolation Forest scores the latest multivariate reading, and the LSTM
reconstructs the recent sequence. Their scores are displayed relative to the
99th percentile of healthy calibration scores. They provide unit-level context;
per-tag notifications come from forecast residuals. None is a failure probability.

Repeated snapshots do not advance persistence. Missing/bad data, stopped units
or cadence gaps clear the affected unit's temporal window and require warmup.
Other units continue normally. A valid but deviating value does not stop any
tag's predictions. This joint model requires all configured tags to be usable;
it does not independently forecast through a missing sensor.

Models remain fixed until explicit retraining. Recent readings update forecast
context, so persistent anomalies can influence later predictions. Validate
sensitivity to slow drift, transitions and correlated faults during the pilot.
No remaining-life estimate, automatic fault diagnosis or plant control is provided.

## PI connection

Enter the HTTPS base URL, Data Archive server and authentication in the app.
Point names resolve through `GET /points?path=\\\\server\\tag`; an optional tag-to-
Web-ID dictionary supports pre-resolved point or attribute streams. Current
values are read using batched `GET /streamsets/value?webId=...` requests. The
connector is read-only and does not create PI points or write predictions to PI.

Supported authentication options depend on what your PI deployment enables:

- Basic over verified HTTPS.
- Windows integrated authentication on a Windows host, using the identity
  running the service; install `requests-negotiate-sspi` separately.
- Bearer token for a gateway that accepts it; there is no token-refresh flow.
- Anonymous only where explicitly enabled by the PI deployment.

Passwords/tokens remain in service memory and must be re-entered after restart.
Server names, username, Web IDs and optional CA-bundle path are stored locally.
The service verifies TLS, optionally using your organization's CA bundle, and
refuses redirects so credentials cannot be redirected. HTTP and inline URL
credentials are rejected. Browser requests are restricted to the loopback
application origins; this service is not a multiuser/authenticated network API.
Do not expose it on a network interface without a proper deployment layer.

`Good=false`, questionable/substituted values, non-numeric digital states,
missing values, stale/future timestamps and excessive timestamp skew are treated
as quality issues. Poll failures clear current results and show a connection
error. Polling resumes automatically at the configured interval. Use the same polling interval as the trained sampling cadence. The service
polls current PI snapshots; it does not backfill archive history. It needs a full
history window after start or a quality reset. Compressed PI tags that do not
receive fresh timestamps can trigger staleness or prevent a new joint snapshot;
verify this against your archive configuration before the live trial.

PI endpoint references: [AVEVA PI Web API reference](https://docs.aveva.com/bundle/pi-web-api-reference/)
and your server's `/piwebapi/help` for its installed version. The connector is
covered by mocked PI tests; connectivity and authentication must be verified
against your server when you enter its settings.

## GPU notebook and model transfer

Open [the training notebook in Colab](https://colab.research.google.com/github/bunjopolo/hydrosense/blob/main/notebooks/hydrosense_training.ipynb),
select a GPU runtime, and follow its cells. It uses the **same Python training
pipeline** as the app, discovers all numeric tags, evaluates later history and
downloads a model package. Use an approved environment for plant data; Colab
requires uploading the file there. CPU training is also supported.

Extract your downloaded package on the work computer. Stop HydroSense, then
activate the model you trained (PowerShell example):

```powershell
.venv\Scripts\python -m hydrosense.cli activate --model C:\path\hydro-model.joblib --trust-model
```

Restart the app and load a replay file or configure PI. Inference runs on CPU,
so the work computer does not need a GPU. Use matching dependency versions on
both machines; the notebook includes a versions manifest. Only load trusted
joblib artifacts: they contain executable Python serialization. Model weights,
learned statistics and tag names should remain private.

## Batch workflow

```bash
PYTHONPATH=src .venv/bin/python -m hydrosense.cli train \
  --data /path/to/healthy.xlsx --sheet Plant \
  --config configs/hydro/assets.json --auto-tags \
  --output artifacts/my-model.joblib --healthy-confirmed

PYTHONPATH=src .venv/bin/python -m hydrosense.cli score \
  --data /path/to/replay.csv --model artifacts/my-model.joblib \
  --output reports/hydro-predictions.jsonl

PYTHONPATH=src .venv/bin/python -m hydrosense.cli activate \
  --model artifacts/my-model.joblib --trust-model
```

Set the timestamp column and timezone in the configuration first. `--auto-tags`
creates one unit from all numeric columns; omit it to use configured unit groups.
Stop the service before activating a CLI/notebook model.

## Checks and limitations

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hydrosense.py -v
cd webapp
npm test
```

Tests exercise real CPU TFT/LSTM training, serialization, chronological split
isolation, independent per-tag notifications, prior forecast timestamps,
CSV/Excel imports, replay logging, PI quality and credential handling. Frontend
checks build and verify the rendered page. The notebook is schema/code validated;
GPU execution and your actual PI server require validation in your environment.
Synthetic tests establish software behaviour, not plant performance.
