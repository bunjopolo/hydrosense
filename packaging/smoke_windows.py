"""Verify the actual EXE's HTTP UI, origin checks and bundled ML runtime on Windows."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request
import urllib.error

exe = Path('dist/HydroSense.exe').resolve()
subprocess.run([str(exe), '--self-test'], check=True, timeout=600)
with tempfile.TemporaryDirectory() as folder:
    with (Path(folder) / 'launcher.log').open('w+') as log:
        process = subprocess.Popen([str(exe), '--no-browser', '--port', '18765', '--data-dir', str(Path(folder) / 'state')], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18765/status', timeout=2) as response:
                        state = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Executable did not start')
                    time.sleep(1)
            assert state['job']['status'] == 'idle'
            with urllib.request.urlopen('http://127.0.0.1:18765/') as response:
                html = response.read().decode()
            assert 'HydroSense' in html and '/assets/' in html
            import re
            script = re.search(r'src="([^"]+\.js)"', html).group(1)
            with urllib.request.urlopen('http://127.0.0.1:18765' + script) as response:
                assert 'javascript' in response.headers['Content-Type']
                assert len(response.read()) > 1000
            bad = urllib.request.Request('http://127.0.0.1:18765/status', headers={'Origin': 'https://untrusted.invalid'})
            try:
                urllib.request.urlopen(bad)
                raise AssertionError('Foreign origin was accepted')
            except urllib.error.HTTPError as exc:
                assert exc.code == 403
            print('PACKAGED_HTTP_TEST_OK')
        finally:
            # onefile uses a parent bootloader and child process; stop both before cleanup.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False)
            process.wait(timeout=30)
