"""One-process desktop launcher: bundled dashboard and ML API on loopback."""
from __future__ import annotations
import argparse
import multiprocessing
from pathlib import Path
import sys
import tempfile
import webbrowser


def self_test():
    """Exercise the frozen ML stack, including dynamically loaded imports and weights."""
    import json
    import joblib
    from .demo import generate
    from .model import train, Monitor
    from .runtime import resource_root
    from .service import State, atomic_json
    from .cli import activate
    from requests_negotiate_sspi import HttpNegotiateAuth
    assert HttpNegotiateAuth
    config = json.loads((resource_root() / 'configs/hydro/assets.json').read_text())
    config.update(lookback=8, horizon=3, epochs=1, hidden_size=8, device='cpu')
    config['assets'] = [{'id': 'test', 'name': 'Packaged model test', 'tags': ['U1.MW', 'U1.GB.TEMP']}]
    frame = generate(320)
    from .selection import selected_frame
    chosen = selected_frame(frame, [[0, 150], [170, 320]], 'timestamp')
    bundle = train(chosen, config, print)
    assert bundle['training_selection']['selected_rows'] == 300
    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / 'model.joblib'
        joblib.dump(bundle, model)
        activate(model, Path(directory) / 'state')
        state = State(Path(directory) / 'state')
        assert state.snapshot()['model']['compatible']
        monitor = state.monitor
        for _, row in frame.iloc[:10].iterrows():
            now = row.timestamp.to_pydatetime()
            values = {tag: float(row[tag]) for tag in config['assets'][0]['tags']}
            result = monitor.score(values, dict.fromkeys(values, now), now, live=False)[0]
        assert len(result['signals']) == 2
        assert all(s['expected'] is not None and len(s['forecast']) == 3 for s in result['signals'])
        assert set(result['components']) == {'isolation_forest', 'lstm_autoencoder'}
    print('PACKAGED_SELF_TEST_OK', flush=True)


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] in ('activate', 'train', 'score'):
        from .cli import main as cli_main
        return cli_main()
    parser = argparse.ArgumentParser(description='HydroSense desktop — leave this window open while monitoring')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    print('Starting HydroSense. Loading the ML libraries may take a moment…', flush=True)
    if args.self_test:
        self_test()
        return
    from http.server import ThreadingHTTPServer
    from .runtime import resource_root, data_directory
    from .service import State, handler_for
    directory = args.data_dir or data_directory()
    static = resource_root() / 'webapp/desktop-dist'
    if not (static / 'index.html').is_file():
        raise RuntimeError('Bundled dashboard is missing. Build it with npm run build:desktop.')
    # Bind before loading state: a second launch must not touch the running app's files.
    server = ThreadingHTTPServer(('127.0.0.1', args.port), lambda *a, **kw: None, bind_and_activate=False)
    try:
        server.server_bind()
        server.server_activate()
        state = State(directory)
        server.RequestHandlerClass = handler_for(state, static)
        url = f'http://127.0.0.1:{server.server_port}/'
        print(f'\nHydroSense is ready: {url}\nModels and settings: {directory}\n'
              'Leave this window open. Close it or press Ctrl+C to stop monitoring.\n', flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            state.stop_event.set()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
