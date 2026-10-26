"""Validation for explicitly trusted HydroSense model files."""
import json
import math

import joblib

from .model import MODEL_VERSION, Monitor, fingerprint, validate_config


def load_trusted_model(path):
    # joblib can execute Python. The caller must obtain trust before calling.
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or bundle.get('version') != MODEL_VERSION:
            raise ValueError('Expected an all-tag HydroSense model. Retrain with this version.')
        config = validate_config(bundle['config'])
        if bundle['config_hash'] != fingerprint(config):
            raise ValueError('Model configuration does not match its saved fingerprint.')
        if not math.isfinite(bundle['sample_seconds']) or bundle['sample_seconds'] <= 0:
            raise ValueError('Invalid model sampling interval.')
        if not isinstance(bundle['created_at'], str) or not isinstance(bundle['reports'], list):
            raise ValueError('Model metadata is missing.')
        units = {a['id']: a for a in config['assets']}
        if set(bundle['models']) != set(units) or {r['id'] for r in bundle['reports']} != set(units):
            raise ValueError('Model units do not match the saved configuration.')
        for report in bundle['reports']:
            if [t['tag'] for t in report['tags']] != units[report['id']]['tags']:
                raise ValueError('Model report tags do not match the saved configuration.')
            for key in ('name', 'fit_rows', 'calibration_rows', 'test_rows', 'test_windows', 'test_flag_rate'):
                report[key]
            for tag in report['tags']:
                for key in ('mae', 'baseline_mae', 'tolerance'):
                    tag[key]
        json.dumps({'config': config, 'reports': bundle['reports']}, allow_nan=False)
        monitor = Monitor(bundle)  # Reconstruct CPU predictors and verify weights.
        return bundle, monitor
    except Exception as exc:
        raise ValueError('Unable to load this HydroSense model. Select a complete .joblib export and use matching training/app dependency versions. ' +
                         (str(exc) if isinstance(exc, ValueError) else 'The file is damaged, incomplete or incompatible.')) from None
