# HydroSense dashboard

Local UI for the Python `hydrosense` service. See the parent README for setup.
Run `../scripts/run_dashboard.sh` from the project to start both services.
UI: http://127.0.0.1:5173. API: http://127.0.0.1:8765.

The browser imports CSV/Excel history, maps arbitrary tags, trains models,
replays held-out readings, and configures PI Web API polling. Training and
scoring happen in Python. Secrets are held only by the local service.
This UI alone is not a hosted ML backend.

`npm test` builds and verifies the rendered page. The existing Vinext/Cloudflare
build scaffold remains available, but no HydroSense deployment is configured.
