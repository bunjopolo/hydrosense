"""Predict every tag from past observations, with independent per-tag alerts."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from numbers import Real

import numpy as np
import pandas as pd

MODEL_VERSION = 2


def normalize_config(config):
    config = copy.deepcopy(config)
    for asset in config.get('assets', []):
        if 'tags' not in asset:
            asset['tags'] = list(dict.fromkeys(asset.pop('inputs', []) + asset.pop('targets', []) +
                                             ([asset['running']['tag']] if asset.get('running') else [])))
    config.setdefault('lookback', 24)
    config.setdefault('horizon', 12)
    config.setdefault('epochs', 10)
    config.setdefault('batch_size', 64)
    config.setdefault('hidden_size', 16)
    config.setdefault('device', 'cpu')
    return config


def validate_config(config):
    if not isinstance(config, dict) or not config.get('timestamp_column'):
        raise ValueError('Set timestamp_column and at least one unit.')
    config = normalize_config(config)
    if not config.get('assets'):
        raise ValueError('Configure at least one unit with tags.')
    ids = set()
    for asset in config['assets']:
        key = asset.get('id', '')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', key) or key in ids:
            raise ValueError('Unit IDs must be unique letters, numbers, hyphens or underscores.')
        ids.add(key)
        tags = asset.get('tags', [])
        if not tags or not all(isinstance(t, str) and t.strip() for t in tags) or len(tags) != len(set(tags)):
            raise ValueError(f'{key}: provide unique numeric tag names.')
        if asset.get('running') and asset['running'].get('tag') not in tags:
            raise ValueError('The optional running tag must also be in the unit tag list.')
    for field, default, lo, hi in [('poll_seconds', 30, 1, 86400), ('stale_seconds', 180, 1, 86400),
                                  ('max_skew_seconds', 120, 0, 86400), ('persistence', 3, 1, 100),
                                  ('lookback', 24, 2, 720), ('horizon', 12, 1, 120),
                                  ('epochs', 10, 1, 1000), ('batch_size', 64, 8, 1024), ('hidden_size', 16, 8, 128)]:
        value = config.get(field, default)
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{field} must be an integer from {lo} to {hi}.')
    if config['hidden_size'] % 2:
        raise ValueError('hidden_size must be even.')
    if config['device'] not in ('cpu', 'cuda', 'auto'):
        raise ValueError('device must be cpu, cuda or auto.')
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(normalize_config(config), sort_keys=True).encode()).hexdigest()


def all_tags(config):
    return list(dict.fromkeys(t for a in normalize_config(config)['assets'] for t in a['tags']))


def numeric_tags(frame, timestamp_column):
    return [str(t) for t in frame.columns if t != timestamp_column and
            np.isfinite(pd.to_numeric(frame[t], errors='coerce').to_numpy(dtype=float)).any()]


def config_for_frame(frame, config, name='Hydro unit'):
    config = normalize_config(config)
    tags = numeric_tags(frame, config['timestamp_column'])
    if not tags:
        raise ValueError('No numeric tag columns were found.')
    config['assets'] = [{'id': 'unit', 'name': name, 'kind': 'Hydro unit', 'tags': tags}]
    cadence = float(frame[config['timestamp_column']].diff().dt.total_seconds().median())
    if not np.isfinite(cadence) or cadence <= 0:
        raise ValueError('At least two distinct timestamps are required.')
    config['poll_seconds'] = min(86400, max(1, round(cadence)))
    return validate_config(config)


def read_csv(text, timestamp_column, source_timezone=None):
    header = next(csv.reader(io.StringIO(text.lstrip('\ufeff'))), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate CSV column names are not supported.')
    frame = pd.read_csv(io.StringIO(text), low_memory=False)
    if timestamp_column not in frame:
        raise ValueError(f'Missing timestamp column: {timestamp_column}')
    raw = frame[timestamp_column].astype(str)
    has_zone = raw.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', case=False)
    if not has_zone.all():
        if not source_timezone or has_zone.any():
            raise ValueError('Use timezone-bearing timestamps, or set source_timezone for an entirely local-time column.')
        try:
            frame[timestamp_column] = pd.to_datetime(raw, errors='raise', format='mixed').dt.tz_localize(source_timezone, ambiguous='raise', nonexistent='raise').dt.tz_convert('UTC')
        except Exception:
            raise ValueError('Invalid local timestamps or timezone, or ambiguous daylight-saving time. Use explicit UTC offsets.') from None
    else:
        frame[timestamp_column] = pd.to_datetime(raw, utc=True, errors='coerce', format='mixed')
    if frame[timestamp_column].isna().any():
        raise ValueError('CSV contains invalid timestamps.')
    if frame[timestamp_column].duplicated().any():
        raise ValueError('CSV contains duplicate timestamps. Resolve them before importing.')
    return frame.sort_values(timestamp_column).reset_index(drop=True)


def train(frame, config, progress=lambda message: None, progress_detail=lambda event: None):
    from .forecasting import fit_stack
    config = validate_config(config)
    missing = set(all_tags(config)) - set(frame.columns)
    if missing:
        raise ValueError('Missing configured tags: ' + ', '.join(sorted(missing)))
    if len(frame) < 250:
        raise ValueError('At least 250 rows are required; use substantially more for pilot evaluation.')
    timestamps = pd.DatetimeIndex(frame[config['timestamp_column']])
    cadence = float(frame.attrs.get('sample_seconds', np.median(np.diff(timestamps.asi8) / 1e9)))
    if cadence <= 0 or not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError('Training timestamps must be ordered and unique.')
    models, reports = {}, []
    last_message = None
    for unit_index, asset in enumerate(config['assets']):
        raw = frame[asset['tags']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        if asset.get('running'):
            index = asset['tags'].index(asset['running']['tag'])
            raw[raw[:, index] < asset['running'].get('min', 0)] = np.nan
        settings = {key: config[key] for key in ('lookback', 'horizon', 'epochs', 'batch_size', 'hidden_size', 'device')}
        settings['sample_seconds'] = cadence
        def report_progress(message, fraction):
            nonlocal last_message
            message = f'{asset.get("name", asset["id"])}: {message}'
            if message != last_message:
                progress(message)
                last_message = message
            # Reserve the final 3% for service serialization and activation.
            progress_detail({'message': message, 'percent': round(97 * (unit_index + round(fraction, 8)) / len(config['assets']), 1)})
        stack, report = fit_stack(raw, timestamps, settings, report_progress)
        models[asset['id']] = stack
        reports.append({'id': asset['id'], 'name': asset.get('name', asset['id']), 'rows': len(frame),
                        **{k: v for k, v in report.items() if k not in ('mae', 'baseline_mae', 'tolerance')},
                        'tags': [{'tag': tag, 'mae': report['mae'][i], 'baseline_mae': report['baseline_mae'][i],
                                  'tolerance': report['tolerance'][i]} for i, tag in enumerate(asset['tags'])]})
    return {'version': MODEL_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(), 'config': config,
            'config_hash': fingerprint(config), 'models': models, 'reports': reports, 'sample_seconds': cadence,
            'training_selection': frame.attrs.get('training_selection')}


class Monitor:
    def __init__(self, bundle, predictor_factory=None):
        if bundle.get('version') != MODEL_VERSION:
            raise ValueError('This model uses the previous input/target workflow. Retrain to forecast all tags.')
        self.bundle = bundle
        if predictor_factory is None:
            from .forecasting import StackPredictor
            predictor_factory = StackPredictor
        self.predictors = {key: predictor_factory(stack) for key, stack in bundle['models'].items()}
        self.windows, self.pending, self.counts, self.last, self.cache = {}, {}, {}, {}, {}

    def score(self, values, times, now, quality=None, live=True):
        config = self.bundle['config']
        output = []
        cadence = self.bundle['sample_seconds']
        origin = pd.Timestamp(now).floor(pd.Timedelta(seconds=cadence)) if live else pd.Timestamp(now)
        for asset in config['assets']:
            key, tags = asset['id'], asset['tags']
            signals = [{'tag': t, 'actual': float(values[t]) if isinstance(values.get(t), Real) and not isinstance(values[t], bool) and np.isfinite(values[t]) else None,
                        'expected': None, 'forecast_issued_at': None, 'residual': None, 'tolerance': None, 'ratio': None,
                        'status': 'warming_up', 'consecutive': 0, 'forecast': []} for t in tags]
            result = {'id': key, 'name': asset.get('name', key), 'kind': asset.get('kind', 'Unit'),
                      'signals': signals, 'timestamp': origin.isoformat(), 'score': None,
                      'status': 'warming_up', 'detail': '', 'components': {}}
            bad = {}
            for signal in signals:
                tag, stamp = signal['tag'], times.get(signal['tag'])
                if signal['actual'] is None or stamp is None or (quality is not None and quality.get(tag) is not True):
                    bad[tag] = 'Missing or bad-quality reading'
                elif live and ((now - stamp).total_seconds() > config.get('stale_seconds', 180) or (stamp - now).total_seconds() > 30):
                    bad[tag] = 'Stale or future PI timestamp'
            observed = [times[t] for t in tags if t in times]
            if observed and (max(observed) - min(observed)).total_seconds() > config.get('max_skew_seconds', 120):
                bad.update(dict.fromkeys(tags, 'Tag timestamps are not aligned'))
            previous = self.last.get(key)
            if previous and origin < previous['origin']:
                bad.update(dict.fromkeys(tags, 'Out-of-order observation'))
            if previous and live and any(times.get(t, previous['times'][t]) < previous['times'][t] for t in tags):
                bad.update(dict.fromkeys(tags, 'Out-of-order PI reading'))
            stopped = asset.get('running') and values.get(asset['running']['tag']) is not None and values[asset['running']['tag']] < asset['running'].get('min', 0)
            if bad or stopped:
                self.windows[key], self.pending[key], self.counts[key] = [], {}, {}
                result['status'] = 'data_quality' if bad else 'not_running'
                result['detail'] = '; '.join(f'{t}: {why}' for t, why in bad.items()) if bad else 'Outside running condition'
                for signal in signals:
                    signal['status'] = 'data_quality' if signal['tag'] in bad else ('not_running' if stopped else 'warming_up')
                self.cache[key] = result
                output.append(result)
                continue
            signature = tuple(times[t] for t in tags)
            if previous and (origin == previous['origin'] or (live and signature == previous['signature'])):
                output.append(copy.deepcopy(self.cache[key]))
                continue
            gap = previous and abs((origin - previous['origin']).total_seconds() - cadence) > max(.001, cadence * .05)
            if gap:
                self.windows[key], self.pending[key], self.counts[key] = [], {}, {}
            self.last[key] = {'origin': origin, 'signature': signature, 'times': dict(times)}
            pending = self.pending.get(key, {}).get(origin.isoformat())
            counts = self.counts.setdefault(key, {})
            if pending is not None:
                expected, tolerance, issued_at = pending
                for i, signal in enumerate(signals):
                    residual = signal['actual'] - float(expected[i])
                    ratio = abs(residual) / float(tolerance[i])
                    counts[signal['tag']] = counts.get(signal['tag'], 0) + 1 if ratio >= 1 else 0
                    signal.update(expected=float(expected[i]), forecast_issued_at=issued_at, residual=residual, tolerance=float(tolerance[i]), ratio=ratio,
                                  consecutive=counts[signal['tag']], status='alert' if counts[signal['tag']] >= config.get('persistence', 3) else ('deviation' if ratio >= 1 else 'normal'))
                result['score'] = max(s['ratio'] for s in signals)
                result['status'] = 'alert' if any(s['status'] == 'alert' for s in signals) else ('deviation' if result['score'] >= 1 else 'normal')
                result['detail'] = 'Each actual value is compared with a forecast issued before this observation.'
            window = self.windows.setdefault(key, [])
            window.append((origin, [s['actual'] for s in signals]))
            window[:] = window[-config['lookback']:]
            if len(window) == config['lookback']:
                q, future_times, components = self.predictors[key].predict([row[1] for row in window], [row[0] for row in window])
                tolerance = self.bundle['models'][key]['tolerance']
                self.pending[key] = {stamp.isoformat(): (q[h, :, 1], tolerance[h], origin.isoformat()) for h, stamp in enumerate(future_times)}
                result['components'] = components
                for i, signal in enumerate(signals):
                    signal['forecast'] = [{'timestamp': stamp.isoformat(), 'expected': float(q[h, i, 1]),
                                           'lower': float(q[h, i, 0]), 'upper': float(q[h, i, 2])} for h, stamp in enumerate(future_times)]
                if pending is None:
                    result['detail'] = 'Forecasts issued for every tag. Waiting for the next actual observation to compare.'
            else:
                result['detail'] = f'Collecting history: {len(window)}/{config["lookback"]} observations.'
            self.cache[key] = copy.deepcopy(result)
            output.append(result)
        return output
