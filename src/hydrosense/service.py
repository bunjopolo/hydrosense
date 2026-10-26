"""Loopback-only local API and polling worker; run with python -m hydrosense.service."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import uuid

import joblib
import pandas as pd

from .data import read_upload
from .demo import generate
from .model import Monitor, all_tags, fingerprint, read_csv, train, validate_config
from .pi import PIClient

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ORIGINS = {f'http://{host}:{port}' for host in ('localhost', '127.0.0.1') for port in (5173, 3000)}


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


class State:
    def __init__(self, directory=None):
        self.directory = Path(directory or ROOT / 'artifacts/hydro')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.config = json.loads((self.directory / 'config.json' if (self.directory / 'config.json').exists() else ROOT / 'configs/hydro/assets.json').read_text())
        self.settings = json.loads((self.directory / 'connection.json').read_text()) if (self.directory / 'connection.json').exists() else {}
        self.secret = ''
        self.bundle = None
        self.job = {'status': 'idle'}
        self.training = None
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
            self.monitor = Monitor(self.bundle)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({'config': self.config, 'connection': {**self.settings, 'has_secret': bool(self.secret)},
                'job': self.job, 'mode': self.mode, 'source': self.source, 'error': self.error,
                'results': self.results, 'history': self.history, 'updated_at': self.updated_at,
                'training_rows': len(self.training) if self.training is not None else 0,
                'replay_rows': len(self.replay) if self.replay is not None else 0, 'replay_index': self.replay_index,
                'model': {'created_at': self.bundle['created_at'], 'reports': self.bundle['reports'],
                          'compatible': fingerprint(self.config) == self.bundle['config_hash']} if self.bundle else None})

    def idle(self):
        if self.mode == 'live' or self.job['status'] == 'training':
            raise ValueError('Stop live monitoring and wait for training before changing this setting.')

    def compatible(self):
        if not self.bundle or self.bundle['config_hash'] != fingerprint(self.config):
            raise ValueError('Train a model for the current asset configuration first.')

    def record(self, results, timestamp):
        self.results = results
        self.updated_at = timestamp
        self.history.append({'timestamp': timestamp, 'assets': results})
        self.history = self.history[-120:]
        self.error = None

    def train_async(self, demo=False):
        self.idle()
        if self.training is None:
            raise ValueError('Import a training CSV first.')
        config, frame = copy.deepcopy(self.config), self.training.copy()
        self.job = {'status': 'training'}
        self.mode = 'idle'
        def work():
            try:
                bundle = train(frame, config)
                name = f'model-{uuid.uuid4().hex}.joblib'
                joblib.dump(bundle, self.directory / name)
                with self.lock:
                    source = 'synthetic' if demo else 'csv'
                    atomic_json(self.directory / 'active.json', {'file': name, 'source': source})
                    self.bundle = bundle
                    self.monitor = Monitor(bundle)
                    self.source = source
                    self.results, self.history = [], []
                    self.job = {'status': 'complete'}
                    self.error = None
            except Exception as exc:
                with self.lock:
                    self.job = {'status': 'failed', 'error': str(exc)}
        threading.Thread(target=work, daemon=True).start()

    def action(self, path, data):
        with self.lock:
            if path == '/config':
                self.idle()
                self.config = validate_config(data)
                atomic_json(self.directory / 'config.json', self.config)
                self.training = self.replay = None
                self.results, self.history = [], []
                self.mode = 'idle'
            elif path == '/csv':
                self.idle()
                frame = read_upload(data, self.config)
                missing = set(all_tags(self.config)) - set(frame.columns)
                # Inspect accepts arbitrary tag headers before the user finishes mapping.
                if data.get('purpose') == 'inspect':
                    return {'columns': list(frame.columns), 'rows': len(frame), 'missing': sorted(missing),
                            'preview': frame.head(5).astype(str).to_dict('records')}
                if missing:
                    raise ValueError('Map CSV columns to assets first. Missing: ' + ', '.join(sorted(missing)))
                if data.get('purpose') == 'replay':
                    self.replay, self.replay_index = frame, 0
                    self.mode = 'idle'
                    self.history, self.results = [], []
                    if self.bundle:
                        self.monitor = Monitor(self.bundle)
                else:
                    self.training = frame
                return {'columns': list(frame.columns), 'rows': len(frame)}
            elif path == '/train':
                if data.get('healthy_confirmed') is not True:
                    raise ValueError('Confirm that training data represents healthy operation.')
                self.train_async()
            elif path == '/demo':
                self.idle()
                self.config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
                atomic_json(self.directory / 'config.json', self.config)
                self.training, self.replay = generate(), generate(360, 83, True, '2026-02-01')
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
                    self.results, self.history = [], []
                for _ in range(max(1, min(int(data.get('steps', 1)), 10))):
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
                allowed = ('url', 'data_server', 'auth', 'username', 'ca_bundle', 'web_ids')
                settings = {k: data[k] for k in allowed if k in data}
                secret = data.get('secret', '')
                # An empty field preserves the in-memory secret only for identical account settings.
                if not secret and all(settings.get(k) == self.settings.get(k) for k in ('url', 'auth', 'username')):
                    secret = self.secret
                PIClient(settings, secret)
                self.settings, self.secret = settings, secret
                atomic_json(self.directory / 'connection.json', settings)
            elif path == '/connection/test':
                self.idle()
                client = PIClient(self.settings, self.secret)
                values, _, quality = client.snapshot(all_tags(self.config))
                return {'message': f'Connection succeeded. {sum(quality.values())}/{len(values)} tags have good numeric values.', 'quality': quality}
            elif path == '/live/start':
                self.idle()
                self.compatible()
                if self.source == 'synthetic':
                    raise ValueError('Train on your healthy plant CSV before starting live PI monitoring.')
                client = PIClient(self.settings, self.secret)
                self.monitor = Monitor(self.bundle)
                self.stop_event = threading.Event()
                self.mode = 'live'
                self.history, self.results, self.error = [], [], None
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
            stop_event.wait(self.config.get('poll_seconds', 30))


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log CSV payloads, credentials or server addresses.

        def allowed(self):
            host = self.headers.get('Host', '').split(':')[0]
            origin = self.headers.get('Origin')
            return host in ('localhost', '127.0.0.1') and (origin is None or origin in ALLOWED_ORIGINS)

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
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            if self.path != '/status':
                return self.send(404, {'error': 'Not found'})
            self.send(200, state.snapshot())

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {'error': 'Origin not allowed'})
            try:
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
