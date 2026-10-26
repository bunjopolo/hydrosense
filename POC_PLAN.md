# HydroSense / SmartSignal pilot

## Implemented

All-tag CSV/Excel discovery, unit grouping, TFT forecasts for every variable,
LSTM and Isolation Forest context, chronological calibration/evaluation, an
all-graphs dashboard, independent in-app tag notifications, replay controls,
read-only PI polling, durable prediction CSVs and a GPU-ready Colab notebook.

## Trial protocol

1. Select one unit and exactly the same tags/engineering units used by its
   SmartSignal model. Export regularly sampled history with explicit timestamps.
2. Have engineers select a healthy period covering normal operating regimes.
   Keep later known events and prospective trial data out of training.
3. Train in the app or notebook. Review each tag's held-out MAE against the
   last-value baseline. Adjust training settings using development history;
   preserve a final untouched test interval for the agreed comparison.
4. Replay labelled events and healthy intervals. Confirm every tag has a graph,
   distinguish data-quality gaps from anomalies, and document expected warmup.
5. Freeze the model/configuration and start read-only PI monitoring alongside
   SmartSignal. Record matching sampling cadence, thresholds, persistence and
   run dates. Verify actual server authentication, quality and compression behaviour.
6. Export HydroSense's pilot CSV and obtain SmartSignal's results separately.
   Align UTC timestamps and exact tag names. HydroSense records the prediction
   issue time as well as the observation time to support a causal comparison.
7. Have engineers label events independently of either model. Compare false
   alerts per unit-day/tag-day, event recall, lead time, duration and useful
   investigations. Report unavailable time/warmup separately and include
   transitions, maintenance and slow drift. Forecast MAE alone is insufficient.

Prediction logs are retained in artifacts/hydro/runs even after a new run starts.
The dashboard export downloads the current run. Notification history is an
in-memory view; transitions can also be reconstructed from the logged states.
There is no SmartSignal connector or automatic side-by-side scoring dashboard;
comparison uses separately exported records and engineer-reviewed event labels.

## Remaining environment validation

Actual plant performance, PI access, GPU execution and acceptable notification
rates require the user's data and environment. Email/SMS, unattended service
packaging, role-based access, model approval UI and plant control are not included.
