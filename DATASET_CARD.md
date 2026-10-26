# HydroSense dataset contract

CSV or .xlsx worksheet: one timestamp column and arbitrary numeric tag columns.
All numeric columns can be detected as one unit; configuration can group tags
into multiple units. Every configured tag is predicted. String metadata is
excluded by automatic detection. Numeric metadata must be removed explicitly
from the unit configuration if it is not a sensor variable.

Use offset-bearing timestamps or configure the source timezone. Headers and
timestamps must be unique. Values remain in source engineering units; PI must
supply matching units and sampling cadence. Use reviewed healthy history for
training and a separate later file for fault replay. At least 250 rows and
complete windows in all chronological splits are required; production assessment
needs substantially more representative data. Missing readings and cadence gaps
exclude affected windows; no future interpolation fills them.

No plant dataset is included. The deterministic synthetic demo has ten tags
covering load, speed, cooling, bearing temperature/vibration, generator temperature,
and transformer loading/temperatures. Replay introduces bearing and transformer
deviations to test the workflow, not to estimate real detection performance.

Browser uploads remain in service memory. Trained artifacts, settings and durable
per-tag prediction logs are stored under ignored artifacts/hydro/. Plant files
and learned models must not be committed to the public repository.
