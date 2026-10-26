Now includes visual training-data selection: review every imported tag, drag to include/exclude periods, zoom, apply the selection, and export included rows for Colab.

Windows x64 executable with the dashboard, CPU machine-learning runtime and Windows PI authentication adapter included.

1. Download **HydroSense.exe** below. No GitHub login, Python, Node.js or administrator installation is required.
2. Double-click it and allow time for the bundled libraries to unpack. The dashboard opens in your default browser.
3. Leave the HydroSense console window open while using the app; close it to stop.

Settings, models and pilot logs are stored under `%LOCALAPPDATA%\HydroSense`, separately from the executable. Existing source-checkout data is not automatically moved. To migrate, close both apps and copy the contents of `artifacts\hydro` into that folder. Keep existing copies until the new app is verified.

This is an unsigned pilot application. If your work computer blocks it, ask IT to approve it; do not disable endpoint protection. No plant data or PI credentials are included.

Packaging simplifies setup; it does not reduce training CPU/RAM requirements. Use the [Colab notebook](https://colab.research.google.com/github/bunjopolo/hydrosense/blob/main/notebooks/hydrosense_training.ipynb) to train elsewhere, then activate your trusted model with `HydroSense.exe activate --model C:\path\hydro-model.joblib --trust-model` while the app is closed.

The executable was built and smoke-tested on a GitHub-hosted Windows runner, including real CPU training, model reload, forecasts and the local HTTP dashboard. Actual plant/PI access still requires your environment.
