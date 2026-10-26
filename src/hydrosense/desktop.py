"""Local Python launcher: built dashboard and ML API on loopback."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import webbrowser


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('activate', 'train', 'score'):
        from .cli import main as cli_main
        return cli_main()
    parser = argparse.ArgumentParser(description='HydroSense desktop — leave this window open while monitoring')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    print('Starting HydroSense. Loading the ML libraries may take a moment…', flush=True)
    from http.server import ThreadingHTTPServer
    from .runtime import resource_root, data_directory
    from .service import State, handler_for
    directory = args.data_dir or data_directory()
    static = resource_root() / 'webapp/desktop-dist'
    if not (static / 'index.html').is_file():
        raise RuntimeError('Built dashboard is missing. Build it with npm run build:desktop.')
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
