"""Loopback-only local API and polling worker; run with python -m hydrosense.service."""
from __future__ import annotations

import argparse
import csv
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import threading
import uuid
from urllib.parse import unquote, urlsplit, parse_qs

import joblib
import pandas as pd

from .data import read_upload
from .artifacts import load_trusted_model
from .demo import generate
from .model import MODEL_VERSION, Monitor, all_tags, config_for_frame, fingerprint, normalize_config, numeric_tags, train, validate_config
from .pi import create_client
from .runtime import data_directory, resource_root
from .selection import normalize_ranges, selected_frame, preview

ROOT = resource_root()
ALLOWED_ORIGINS = {f'http://{host}:{port}' for host in ('localhost', '127.0.0.1') for port in (5173, 3000)}
MAX_MODEL_BYTES = 512 * 1024 * 1024


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


class State:
    def __init__(self, directory=None):
        self.directory = Path(directory or data_directory())
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.config = json.loads((self.directory / 'config.json' if (self.directory / 'config.json').exists() else ROOT / 'configs/hydro/assets.json').read_text())
        self.config = normalize_config(self.config)
        self.events = []
        self.signal_states = {}
        self.logged_stamps = {}
        self.run_path = None
        self.migration_note = None
        self.settings = json.loads((self.directory / 'connection.json').read_text()) if (self.directory / 'connection.json').exists() else {}
        self.secret = ''
        self.bundle = None
        self.job = {'status': 'idle'}
        self.training = None
        self.training_ranges = []
        self.training_revision = None
        self.training_source = 'csv'
        self.replay = None
        self.replay_index = 0
        self.mode = 'idle'
        self.source = 'none'
        self.results = []
        self.history = []
        self.error = None
        self.monitor = None
        self.stop_event = threading.Event()
        self.worker = None
        self.updated_at = None
        pointer = self.directory / 'active.json'
        if pointer.exists():
            metadata = json.loads(pointer.read_text())
            name = metadata['file']
            if Path(name).name != name or not name.endswith('.joblib'):
                raise ValueError('Invalid local model manifest.')
            self.bundle = joblib.load(self.directory / name)  # Only service-owned local files.
            self.source = metadata.get('source', 'csv')
            if self.bundle.get('version') == MODEL_VERSION:
                self.monitor = Monitor(self.bundle)
            else:
                self.bundle = None
                self.migration_note = 'The earlier model predicted selected targets. Retrain to forecast every tag.'

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({'config': self.config, 'connection': {**self.settings, 'has_secret': bool(self.secret)},
                'job': self.job, 'mode': self.mode, 'source': self.source, 'error': self.error,
                'results': self.results, 'history': self.history, 'updated_at': self.updated_at,
                'events': self.events, 'migration_note': self.migration_note, 'has_export': self.run_path is not None,
                'sample_seconds': self.bundle.get('sample_seconds') if self.bundle else None,
                'training_rows': len(self.training) if self.training is not None else 0,
                'training_revision': self.training_revision, 'training_ranges': self.training_ranges,
                'selected_training_rows': sum(end - start for start, end in self.training_ranges),
                'replay_rows': len(self.replay) if self.replay is not None else 0, 'replay_index': self.replay_index,
                'model': {'created_at': self.bundle['created_at'], 'reports': self.bundle['reports'],
                          'compatible': fingerprint(self.config) == self.bundle['config_hash']} if self.bundle else None})

    def idle(self):
        if self.mode == 'live' or self.job['status'] == 'training':
            raise ValueError('Stop live monitoring and wait for training before changing this setting.')

    def compatible(self):
        if not self.bundle or self.bundle['config_hash'] != fingerprint(self.config):
            raise ValueError('Train a model for the current asset configuration first.')

    def reset_observations(self):
        self.results, self.history, self.events = [], [], []
        self.signal_states, self.logged_stamps = {}, {}
        self.updated_at, self.run_path = None, None

    def import_model(self, stream, length, trusted=False):
        with self.lock:
            self.idle()
            if trusted is not True:
                raise ValueError('Confirm that you trained or trust this model before loading it.')
            if not 0 < length <= MAX_MODEL_BYTES:
                raise ValueError('Choose a model between 1 byte and 512 MB. Use CLI activation for larger models.')
            name = f'model-{uuid.uuid4().hex}.joblib'
            staged = self.directory / (name + '.upload')
            target = self.directory / name
            try:
                # Stream directly to disk: no base64 copy or browser-sized memory buffer.
                remaining = length
                with staged.open('wb') as output:
                    while remaining:
                        chunk = stream.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Model upload was interrupted. Select the file again.')
                        output.write(chunk)
                        remaining -= len(chunk)
                bundle, monitor = load_trusted_model(staged)
                staged.replace(target)
                try:
                    atomic_json(self.directory / 'config.json', bundle['config'])
                    atomic_json(self.directory / 'active.json', {'file': name, 'source': 'notebook'})
                except Exception:
                    atomic_json(self.directory / 'config.json', self.config)
                    raise
            except Exception:
                target.unlink(missing_ok=True)
                raise
            finally:
                staged.unlink(missing_ok=True)
            self.config, self.bundle, self.monitor = bundle['config'], bundle, monitor
            self.source, self.mode = 'notebook', 'idle'
            self.training = self.replay = None
            self.training_ranges, self.training_revision = [], None
            self.training_source, self.replay_index = 'csv', 0
            self.reset_observations()
            self.migration_note = self.error = None
            self.job = {'status': 'idle'}
            unit_count = len(self.config['assets'])
            return {'message': f'Model loaded: {unit_count} {"unit" if unit_count == 1 else "units"}, {len(all_tags(self.config))} tags. Ready for replay or live PI.'}

    def record(self, results, timestamp):
        self.results, self.updated_at = results, timestamp
        # History holds actual and previously issued predictions; future arrays live only in the current result.
        compact = copy.deepcopy(results)
        for asset in compact:
            for signal in asset['signals']:
                signal.pop('forecast', None)
        if not self.history or [a['timestamp'] for a in self.history[-1]['assets']] != [a['timestamp'] for a in compact]:
            self.history.append({'timestamp': timestamp, 'assets': compact})
            self.history = self.history[-360:]
        rows = []
        for asset in results:
            for signal in asset['signals']:
                key = (asset['id'], signal['tag'])
                status = signal['status']
                before = self.signal_states.get(key)
                if status != before and (status in ('deviation', 'alert', 'data_quality') or (status == 'normal' and before in ('deviation', 'alert'))):
                    self.events.append({'id': uuid.uuid4().hex, 'timestamp': timestamp, 'unit': asset['name'],
                                        'tag': signal['tag'], 'status': status, 'actual': signal['actual'],
                                        'expected': signal['expected'], 'ratio': signal['ratio']})
                    self.events = self.events[-200:]
                self.signal_states[key] = status
                stamp = asset['timestamp']
                if self.logged_stamps.get(key) == stamp:
                    continue
                self.logged_stamps[key] = stamp
                rows.append({'timestamp': stamp, 'unit': asset['name'], 'tag': signal['tag'],
                             'actual': signal['actual'], 'predicted': signal['expected'],
                             'forecast_issued_at': signal.get('forecast_issued_at'),
                             'residual': signal['residual'], 'deviation_ratio': signal['ratio'],
                             'calibrated_band': signal['tolerance'], 'status': status,
                             'consecutive': signal['consecutive'], 'model_created_at': self.bundle['created_at'],
                             'source': 'pi' if self.mode == 'live' else 'replay',
                             'isolation_forest': asset.get('components', {}).get('isolation_forest'),
                             'lstm_autoencoder': asset.get('components', {}).get('lstm_autoencoder')})
        if rows:
            if self.run_path is None:
                runs = self.directory / 'runs'
                runs.mkdir(exist_ok=True)
                self.run_path = runs / f'predictions-{uuid.uuid4().hex}.csv'
            exists = self.run_path.exists()
            with self.run_path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                if not exists:
                    writer.writeheader()
                writer.writerows(rows)
        self.error = None

    def train_async(self, demo=False):
        self.idle()
        if self.training is None:
            raise ValueError('Import a training CSV first.')
        config = copy.deepcopy(self.config)
        frame = selected_frame(self.training, self.training_ranges, config['timestamp_column'])
        if len(frame) < 250:
            raise ValueError('Select at least 250 training rows in sufficiently long continuous ranges.')
        self.job = {'status': 'training', 'percent': 0, 'message': 'Preparing training…'}
        self.mode = 'idle'
        def work():
            try:
                def progress(event):
                    with self.lock:
                        self.job = {'status': 'training', **event}
                bundle = train(frame, config, progress_detail=progress)
                progress({'percent': 97, 'message': 'Saving trained models…'})
                name = f'model-{uuid.uuid4().hex}.joblib'
                joblib.dump(bundle, self.directory / name)
                progress({'percent': 99, 'message': 'Loading models for prediction…'})
                monitor = Monitor(bundle)
                with self.lock:
                    source = 'synthetic' if demo or self.training_source == 'synthetic' else 'csv'
                    atomic_json(self.directory / 'active.json', {'file': name, 'source': source})
                    self.bundle = bundle
                    self.monitor = monitor
                    self.source = source
                    self.reset_observations()
                    self.migration_note = None
                    self.job = {'status': 'complete', 'percent': 100, 'message': 'Training complete. Models ready.'}
                    self.error = None
            except Exception as exc:
                with self.lock:
                    self.job = {**self.job, 'status': 'failed', 'error': str(exc)}
        threading.Thread(target=work, daemon=True).start()

    def action(self, path, data):
        with self.lock:
            if path == '/config':
                self.idle()
                self.config = validate_config(data)
                atomic_json(self.directory / 'config.json', self.config)
                self.training = self.replay = None
                self.training_ranges, self.training_revision = [], None
                self.reset_observations()
                self.mode = 'idle'
            elif path == '/csv':
                self.idle()
                frame = read_upload(data, self.config)
                missing = set(all_tags(self.config)) - set(frame.columns)
                # Inspect accepts arbitrary tag headers before the user finishes mapping.
                if data.get('purpose') == 'inspect':
                    return {'columns': list(frame.columns), 'numeric_columns': numeric_tags(frame, data.get('timestamp_column') or self.config['timestamp_column']), 'rows': len(frame), 'missing': sorted(missing),
                            'preview': frame.head(5).astype(str).to_dict('records')}
                if data.get('auto_configure') and data.get('purpose') != 'replay':
                    candidate = {**self.config, 'timestamp_column': data.get('timestamp_column') or self.config['timestamp_column'],
                                 'source_timezone': data.get('source_timezone', self.config.get('source_timezone', 'UTC'))}
                    self.config = config_for_frame(frame, candidate, data.get('unit_name') or 'Hydro unit')
                    atomic_json(self.directory / 'config.json', self.config)
                    self.reset_observations()
                    self.replay = None
                    missing = set()
                if missing:
                    raise ValueError('File is missing model tags: ' + ', '.join(sorted(missing)))
                if data.get('purpose') == 'replay':
                    self.replay, self.replay_index = frame, 0
                    self.mode = 'idle'
                    self.reset_observations()
                    if self.bundle:
                        self.monitor = Monitor(self.bundle)
                else:
                    self.training = frame
                    self.training_ranges = [[0, len(frame)]]
                    self.training_revision = uuid.uuid4().hex
                    self.training_source = 'csv'
                return {'columns': list(frame.columns), 'rows': len(frame)}
            elif path in ('/training/preview', '/training/selection'):
                if self.training is None:
                    raise ValueError('Import training data first.')
                if data.get('revision') != self.training_revision:
                    raise ValueError('Training data changed. Reload the review before selecting rows.')
                if path == '/training/selection':
                    self.idle()
                    self.training_ranges = normalize_ranges(data.get('ranges'), len(self.training))
                    return {'ranges': self.training_ranges, 'selected_rows': sum(b - a for a, b in self.training_ranges)}
                return {**preview(self.training, self.config['timestamp_column'],
                                  numeric_tags(self.training, self.config['timestamp_column']),
                                  data.get('start', 0), data.get('end')),
                        'revision': self.training_revision, 'rows': len(self.training)}
            elif path == '/train':
                if data.get('healthy_confirmed') is not True:
                    raise ValueError('Confirm that training data represents healthy operation.')
                if 'revision' in data and data['revision'] != self.training_revision:
                    raise ValueError('Training data changed. Review it again before training.')
                self.train_async()
            elif path == '/demo':
                self.idle()
                self.config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
                atomic_json(self.directory / 'config.json', self.config)
                self.config['epochs'] = 2
                atomic_json(self.directory / 'config.json', self.config)
                self.training, self.replay = generate(1200), generate(360, 83, True, '2026-02-01')
                self.training_ranges = [[0, len(self.training)]]
                self.training_revision = uuid.uuid4().hex
                self.training_source = 'synthetic'
                self.replay_index = 0
                self.train_async(demo=True)
            elif path == '/replay':
                self.idle()
                self.compatible()
                if self.replay is None:
                    raise ValueError('Import a replay CSV or load the synthetic demo.')
                if data.get('reset'):
                    self.replay_index = 0
                    self.monitor = Monitor(self.bundle)
                    self.reset_observations()
                if 'seek' in data:
                    target = max(0, min(int(data['seek']), len(self.replay) - 1))
                    self.replay_index = max(0, target - self.config['lookback'] - 120)
                    self.monitor = Monitor(self.bundle)
                    self.reset_observations()
                    steps = target - self.replay_index + 1
                else:
                    steps = max(1, min(int(data.get('steps', 1)), 10))
                for _ in range(steps):
                    if self.replay_index >= len(self.replay):
                        break
                    row = self.replay.iloc[self.replay_index]
                    now = row[self.config['timestamp_column']].to_pydatetime()
                    values = {t: pd.to_numeric(row[t], errors='coerce') for t in all_tags(self.config)}
                    times = dict.fromkeys(values, now)
                    self.record(self.monitor.score(values, times, now, live=False), now.isoformat())
                    self.replay_index += 1
                self.mode = 'replay'
            elif path == '/connection':
                self.idle()
                allowed = ('connector', 'url', 'data_server', 'auth', 'username', 'ca_bundle', 'web_ids')
                settings = {k: data[k] for k in allowed if k in data}
                secret = data.get('secret', '')
                if settings.get('connector') == 'piconnect':
                    settings = {'connector': 'piconnect', 'data_server': settings.get('data_server', '')}
                    secret = ''
                # An empty field preserves the in-memory secret only for identical account settings.
                if settings.get('connector') != 'piconnect' and not secret and all(settings.get(k) == self.settings.get(k) for k in ('connector', 'url', 'auth', 'username')):
                    secret = self.secret
                create_client(settings, secret)
                self.settings, self.secret = settings, secret
                atomic_json(self.directory / 'connection.json', settings)
            elif path == '/connection/test':
                self.idle()
                client = create_client(self.settings, self.secret)
                values, _, quality = client.snapshot(all_tags(self.config))
                return {'message': f'Connection succeeded. {sum(quality.values())}/{len(values)} tags have good numeric values.', 'quality': quality}
            elif path == '/live/start':
                self.idle()
                self.compatible()
                if self.source == 'synthetic':
                    raise ValueError('Train on your healthy plant CSV before starting live PI monitoring.')
                cadence = self.bundle['sample_seconds']
                if abs(self.config.get('poll_seconds', 30) - cadence) > max(.001, cadence * .05):
                    raise ValueError(f'PI polling must match the trained sampling cadence ({cadence:g} seconds). Update the configuration and retrain.')
                client = create_client(self.settings, self.secret)
                self.monitor = Monitor(self.bundle)
                self.stop_event = threading.Event()
                self.mode = 'live'
                self.reset_observations()
                self.error = None
                self.worker = threading.Thread(target=self.poll, args=(client, self.stop_event), daemon=True)
                self.worker.start()
            elif path == '/live/stop':
                self.stop_event.set()
                self.mode = 'idle'
            else:
                raise ValueError('Unknown action.')
        return {'ok': True}

    def poll(self, client, stop_event):
        while not stop_event.is_set():
            try:
                values, times, quality = client.snapshot(all_tags(self.bundle['config']))
                now = datetime.now(timezone.utc)
                with self.lock:
                    if stop_event.is_set():
                        return
                    self.record(self.monitor.score(values, times, now, quality), now.isoformat())
            except Exception as exc:
                with self.lock:
                    if stop_event.is_set():
                        return
                    self.error = str(exc)
                    self.results = []
                    self.monitor = Monitor(self.bundle)
            # Anchor polls to trained time buckets; inference/network time must not
            # accumulate into skipped samples and repeated temporal warmups.
            cadence = self.bundle['sample_seconds']
            delay = cadence - datetime.now(timezone.utc).timestamp() % cadence
            stop_event.wait(delay)


def handler_for(state, static_directory=None):
    static_root = Path(static_directory).resolve() if static_directory else None
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log CSV payloads, credentials or server addresses.

        def allowed(self):
            host = self.headers.get('Host', '').split(':')[0]
            origin = self.headers.get('Origin')
            local_origins = {f'http://{name}:{self.server.server_port}' for name in ('localhost', '127.0.0.1')}
            return host in ('localhost', '127.0.0.1') and (origin is None or origin in ALLOWED_ORIGINS or (static_root and origin in local_origins))

        def send(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            if self.headers.get('Origin') in ALLOWED_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
                self.send_header('Vary', 'Origin')
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin', 'http://localhost:5173'))
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-HydroSense-Trust-Model')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            path = unquote(urlsplit(self.path).path)
            if path == '/export/training':
                with state.lock:
                    revision = parse_qs(urlsplit(self.path).query).get('revision', [None])[0]
                    if state.training is None or revision != state.training_revision:
                        return self.send(409, {'error': 'Training data changed. Review it again before exporting.'})
                    frame = selected_frame(state.training, state.training_ranges, state.config['timestamp_column'])
                    if frame.empty:
                        return self.send(400, {'error': 'No training rows are included.'})
                    payload = frame.to_csv(index=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename=hydrosense-selected-training.csv')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if static_root and (path == '/' or path.startswith('/assets/')):
                file = (static_root / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
                if not file.is_relative_to(static_root) or not file.is_file():
                    return self.send(404, {'error': 'Not found'})
                payload = file.read_bytes()
                self.send_response(200)
                kind = 'text/javascript' if file.suffix == '.js' else mimetypes.guess_type(str(file))[0]
                self.send_header('Content-Type', kind or 'application/octet-stream')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path == '/export/predictions':
                with state.lock:
                    payload = state.run_path.read_bytes() if state.run_path else None
                if payload is None:
                    return self.send(404, {'error': 'No predictions have been logged yet.'})
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename=hydrosense-predictions.csv')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path != '/status':
                return self.send(404, {'error': 'Not found'})
            self.send(200, state.snapshot())

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            try:
                if self.path == '/model/import':
                    if self.headers.get('Content-Type') != 'application/octet-stream':
                        raise ValueError('Expected a binary .joblib model file.')
                    self.connection.settimeout(60)
                    self.send(200, state.import_model(self.rfile, int(self.headers.get('Content-Length', '0')),
                                                     trusted=self.headers.get('X-HydroSense-Trust-Model') == 'true'))
                    return
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 64 * 1024 * 1024:
                    raise ValueError('Request must be between 1 byte and 64 MB. Use the CLI for larger CSVs.')
                data = json.loads(self.rfile.read(length))
                self.send(200, state.action(self.path, data))
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {'error': str(exc)})
            except Exception:
                self.send(500, {'error': 'Local operation failed. Check file access and configuration.'})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    state = State()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(state))
    print(f'HydroSense local API: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        server.server_close()


if __name__ == '__main__':
    main()
