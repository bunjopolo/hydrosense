# HydroSense

Local machine-learning condition monitoring for hydro generators, bearings,
transformers and other assets. Import CSV or Excel history with **any numeric tag
names and any configured number of tags**, choose the operating inputs and tags
to predict for each asset, then train. The same saved models score file replay
and incoming PI Web API snapshots.

Expected values are learned from your file, not hard-coded. They change with the
incoming operating inputs. Retraining on different healthy history changes the
learned relationships. This is conditional **expected behaviour at the current
time**, not a future forecast, remaining-life estimate or fault diagnosis.

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

1. **Try the synthetic demo** on Monitor. It trains real models on generated
   data for a bearing, generator and transformer, with deviations in replay.
   Click Play replay. Demo values are not plant data. The demo replaces the
   active configuration/model; previous model files are retained on disk.
2. **Data & training:** choose your timestamp column, source timezone for
   timestamps without offsets, and optionally an Excel sheet. Import a `.csv`
   or `.xlsx` file and inspect its columns. Browser uploads are limited to
   45 MB; the batch CLI handles larger files subject to available memory.
3. Select operating **inputs** and **predicted tags**, name the asset, and
   replace the example assets with your mapping. Add further assets as needed.
   An input cannot also be a predicted tag within that asset.
4. Click **Use file for training**, confirm this is known healthy operation,
   and train. Review the held-out errors, flagged fraction and unseen operating
   range fraction. Fault labels are not needed, but training on degraded data
   will teach the model that degradation is normal.
5. Import a separate file for replay. Check expected-versus-actual traces and
   the residuals in the tags' original units. Use historical known faults to
   assess detection and false alarms before live use.
6. **PI connection:** enter server settings locally, save them, test access to
   the configured tags, then start live prediction. You do not need to share
   server details with anyone. Synthetic models cannot be started against PI.

Configuration changes clear imported files and invalidate the active model
until retraining. Import/replay buffers are in memory and disappear on restart.
Models and non-secret configuration persist in ignored `artifacts/hydro/`.

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

Each asset needs at least **250 complete, running, healthy rows**, at least one
input and one target. This minimum is only an execution requirement, not enough
evidence of operational model quality. Use substantially more history covering
the intended operating envelope. Use comparable, synchronized sampling cadence
and identical engineering units in training and PI. There is no application
limit on tag count, but memory, training time, request size and PI capacity apply.

## Modelling and alerts

Each asset gets an Extra Trees regression model. Input/output columns are
persisted with the model and aligned by exact name, independent of column order.
The earliest 60% of eligible rows fit the estimator, the next 20% calibrate a
per-target 99th-percentile absolute residual band, and the latest 20% evaluate it.
Only the fit split determines the learned operating ranges. The band has a small
floor for constant signals. No held-out data fits the estimator or thresholds.

A tag's deviation is `abs(actual - expected) / calibrated_band`. At 1× or above,
it exceeds that band. An asset's score is its largest tag deviation; it is not a
probability. With many targets, the asset-level flagged fraction can exceed 1%.
The observed held-out flagged fraction is reported, not an accuracy claim.

Three consecutive **fresh** deviations produce a persistent-deviation alert by
default. Repeated snapshots do not advance the count. Bad data, stopped states,
normal observations, or long gaps reset persistence. Configurable running gates
suppress training/scoring below an operating threshold. Inputs beyond their
learned ranges plus a 5% margin produce an unseen-operation state. This range
check does not detect every unseen combination of otherwise familiar inputs.

The current baseline uses contemporaneous operating inputs, not temporal lag
features or a thermal dynamics model. Thermal lag, transitions and maintenance
regimes require careful operating-window selection and prospective validation.
Independent protection and existing plant alarms remain necessary.

See `configs/hydro/assets.json` for the illustrative configuration. Asset names,
IDs, kinds, input tags, target tags, running gates, polling interval, staleness,
skew and persistence can be edited in **Assets & tags**.

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
error. Polling resumes automatically at the configured interval. Only the last
120 observations are retained for the dashboard; durable event storage and
notification integrations are not implemented.

PI endpoint references: [AVEVA PI Web API reference](https://docs.aveva.com/bundle/pi-web-api-reference/)
and your server's `/piwebapi/help` for its installed version. The connector is
covered by mocked PI tests; connectivity and authentication must be verified
against your server when you enter its settings.

## Batch workflow

```bash
PYTHONPATH=src .venv/bin/python -m hydrosense.cli train \
  --data /path/to/healthy.xlsx --sheet Plant \
  --config configs/hydro/assets.json \
  --output artifacts/my-model.joblib --healthy-confirmed

PYTHONPATH=src .venv/bin/python -m hydrosense.cli score \
  --data /path/to/replay.csv --model artifacts/my-model.joblib \
  --output reports/hydro-predictions.jsonl
```

The CLI output is a standalone model; training in the dashboard activates its
model for the local service. Only load locally trusted joblib files; model
serialization is executable Python data. Model-file upload is not exposed by
the app.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_hydrosense.py -v
cd webapp
npm test
```

Backend coverage includes arbitrary tags, data-driven prediction changes,
retraining changes, target leakage prevention, chronological split isolation,
Excel/local timezone imports, persistence, PI quality/staleness, serialization,
service lifecycle and credential persistence checks.
