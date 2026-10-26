"""One conditional normal-behaviour model per asset, shared by replay and PI."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from numbers import Real
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


def validate_config(config):
    if not isinstance(config, dict) or not config.get('timestamp_column'):
        raise ValueError('Set timestamp_column and at least one asset.')
    assets = config.get('assets', [])
    if not assets:
        raise ValueError('Configure at least one asset.')
    ids = set()
    for asset in assets:
        key = asset.get('id', '')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', key) or key in ids:
            raise ValueError('Asset IDs must be unique letters, numbers, hyphens or underscores.')
        ids.add(key)
        for field in ('inputs', 'targets'):
            tags = asset.get(field, [])
            if not tags or not all(isinstance(t, str) and t.strip() for t in tags) or len(tags) != len(set(tags)):
                raise ValueError(f'{key}: provide unique {field} tags.')
        if set(asset['inputs']) & set(asset['targets']):
            raise ValueError(f'{key}: a target cannot also be an input (target leakage).')
        gate = asset.get('running')
        if gate and (not gate.get('tag') or not np.isfinite(float(gate.get('min', 0)))):
            raise ValueError(f'{key}: running needs a tag and finite minimum.')
    for field, default, lo, hi in [('poll_seconds', 30, 1, 3600), ('stale_seconds', 180, 1, 86400),
                                  ('max_skew_seconds', 120, 0, 86400), ('persistence', 3, 1, 100)]:
        value = config.get(field, default)
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{field} must be an integer from {lo} to {hi}.')
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def all_tags(config):
    return list(dict.fromkeys(t for a in config['assets'] for t in
        a['inputs'] + a['targets'] + ([a['running']['tag']] if a.get('running') else [])))


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


def train(frame, config):
    validate_config(config)
    missing = set(all_tags(config)) - set(frame.columns)
    if missing:
        raise ValueError('Missing configured tags: ' + ', '.join(sorted(missing)))
    models, reports = {}, []
    for asset in config['assets']:
        tags = asset['inputs'] + asset['targets']
        if asset.get('running'):
            tags = list(dict.fromkeys(tags + [asset['running']['tag']]))
        numeric = frame[tags].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid = numeric.notna().all(axis=1)
        if asset.get('running'):
            valid &= numeric[asset['running']['tag']] >= asset['running'].get('min', 0)
        rows = numeric.loc[valid]
        if len(rows) < 250:
            raise ValueError(f"{asset['id']}: need at least 250 complete, running, healthy rows; found {len(rows)}.")
        n_fit, n_cal = int(len(rows) * .6), int(len(rows) * .8)
        fit, cal, test = rows.iloc[:n_fit], rows.iloc[n_fit:n_cal], rows.iloc[n_cal:]
        model = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=1)
        x, y = asset['inputs'], asset['targets']
        model.fit(fit[x].to_numpy(), fit[y].to_numpy().ravel() if len(y) == 1 else fit[y].to_numpy())
        def predict(part):
            return model.predict(part[x].to_numpy()).reshape(len(part), len(y))
        cal_error = np.abs(cal[y].to_numpy() - predict(cal))
        # Freeze normal bounds using calibration only; reserve the final block for evaluation.
        tolerance = np.maximum(np.quantile(cal_error, .99, axis=0), np.maximum(fit[y].std().to_numpy() * .01, 1e-6))
        test_error = np.abs(test[y].to_numpy() - predict(test))
        low, high = fit[x].min().to_numpy(), fit[x].max().to_numpy()
        margin = np.maximum((high - low) * .05, 1e-6)
        test_x = test[x].to_numpy()
        in_domain = ((test_x >= low - margin) & (test_x <= high + margin)).all(axis=1)
        report = {
            'id': asset['id'], 'name': asset.get('name', asset['id']), 'rows': len(rows),
            'excluded_rows': len(frame) - len(rows), 'fit_rows': len(fit), 'calibration_rows': len(cal),
            'test_rows': len(test), 'test_flag_rate': float((test_error / tolerance >= 1).any(axis=1).mean()),
            'test_out_of_domain_rate': float((~in_domain).mean()),
            'tags': [{'tag': tag, 'mae': float(test_error[:, i].mean()), 'tolerance': float(tolerance[i])} for i, tag in enumerate(y)],
            'start': frame.loc[valid, config['timestamp_column']].iloc[0].isoformat(),
            'end': frame.loc[valid, config['timestamp_column']].iloc[-1].isoformat(),
        }
        models[asset['id']] = {'estimator': model, 'tolerance': tolerance, 'low': low - margin, 'high': high + margin}
        reports.append(report)
    return {'version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'config': config,
            'config_hash': fingerprint(config), 'models': models, 'reports': reports}


def evaluate(bundle, values):
    output = []
    for asset in bundle['config']['assets']:
        base = {'id': asset['id'], 'name': asset.get('name', asset['id']), 'kind': asset.get('kind', 'Asset'), 'signals': []}
        needed = asset['inputs'] + asset['targets'] + ([asset['running']['tag']] if asset.get('running') else [])
        bad = [t for t in needed if not isinstance(values.get(t), Real) or isinstance(values.get(t), bool) or not np.isfinite(values[t])]
        if bad:
            output.append({**base, 'status': 'data_quality', 'detail': 'Missing or invalid: ' + ', '.join(bad), 'score': None})
            continue
        if asset.get('running') and values[asset['running']['tag']] < asset['running'].get('min', 0):
            output.append({**base, 'status': 'not_running', 'detail': 'Outside configured running condition', 'score': None})
            continue
        model = bundle['models'][asset['id']]
        x = np.array([values[t] for t in asset['inputs']])
        expected = np.asarray(model['estimator'].predict(x.reshape(1, -1))).reshape(-1)
        actual = np.array([values[t] for t in asset['targets']])
        ratios = np.abs(actual - expected) / model['tolerance']
        signals = [{'tag': tag, 'actual': float(actual[i]), 'expected': float(expected[i]),
                    'residual': float(actual[i] - expected[i]), 'tolerance': float(model['tolerance'][i]),
                    'ratio': float(ratios[i])} for i, tag in enumerate(asset['targets'])]
        outside = [t for i, t in enumerate(asset['inputs']) if x[i] < model['low'][i] or x[i] > model['high'][i]]
        output.append({**base, 'status': 'out_of_domain' if outside else ('deviation' if ratios.max() >= 1 else 'normal'),
                       'detail': 'Unseen operating range: ' + ', '.join(outside) if outside else 'Expected behaviour conditioned on operating inputs',
                       'score': None if outside else float(ratios.max()), 'signals': signals})
    return output


class Monitor:
    def __init__(self, bundle):
        self.bundle = bundle
        self.counts = {}
        self.last_times = {}

    def score(self, values, times, now, quality=None, live=True):
        config = self.bundle['config']
        checked = dict(values)
        reasons = {}
        for tag in all_tags(config):
            timestamp = times.get(tag)
            if quality and quality.get(tag) is not True:
                reasons[tag] = 'bad PI quality'
            elif timestamp is None:
                reasons[tag] = 'missing timestamp'
            elif live and ((now - timestamp).total_seconds() > config.get('stale_seconds', 180) or (timestamp - now).total_seconds() > 30):
                reasons[tag] = 'stale or future timestamp'
            if tag in reasons:
                checked[tag] = None
        results = evaluate(self.bundle, checked)
        for result, asset in zip(results, config['assets']):
            tags = asset['inputs'] + asset['targets'] + ([asset['running']['tag']] if asset.get('running') else [])
            observed = [times[t] for t in tags if t in times]
            if observed and (max(observed) - min(observed)).total_seconds() > config.get('max_skew_seconds', 120):
                result.update(status='data_quality', score=None, signals=[], detail='Tag timestamps are not aligned')
            problems = [f'{t}: {reasons[t]}' for t in tags if t in reasons]
            if problems:
                result['detail'] = '; '.join(problems)
            key = asset['id']
            stamp = min(observed) if len(observed) == len(tags) else None
            previous = self.last_times.get(key)
            fresh = stamp is not None and (previous is None or stamp > previous)
            if stamp is not None and previous is not None and stamp < previous:
                result.update(status='data_quality', score=None, signals=[], detail='Out-of-order observation')
            if result['status'] != 'deviation':
                self.counts[key] = 0
            elif fresh:
                if previous and (stamp - previous).total_seconds() > config.get('stale_seconds', 180):
                    self.counts[key] = 0
                self.counts[key] = self.counts.get(key, 0) + 1
            if fresh:
                self.last_times[key] = stamp
            result['consecutive'] = self.counts.get(key, 0)
            if result['status'] == 'deviation' and result['consecutive'] >= config.get('persistence', 3):
                result['status'] = 'alert'
            result['timestamp'] = stamp.isoformat() if stamp else None
        return results
