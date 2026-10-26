# Build on Windows. Torch is installed from its CPU wheel index by the workflow.
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_all, copy_metadata

sys.setrecursionlimit(5000)
root = Path(SPECPATH).parent

datas = [(str(root / 'webapp/desktop-dist'), 'webapp/desktop-dist'),
         (str(root / 'configs/hydro/assets.json'), 'configs/hydro')]
binaries, hiddenimports = [], ['requests_negotiate_sspi', 'win32timezone']
for package in ['pytorch_forecasting', 'lightning', 'lightning_fabric', 'torchmetrics']:
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
for distribution in ['pytorch-forecasting', 'requests-negotiate-sspi']:
    datas += copy_metadata(distribution, recursive=True)

a = Analysis([str(root / 'packaging/launch.py')], pathex=[str(root / 'src')],
             binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             excludes=['IPython', 'notebook', 'pytest'],
             module_collection_mode={'pytorch_forecasting': 'pyz+py', 'lightning': 'pyz+py'})
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='HydroSense',
          console=True, debug=False, strip=False, upx=False)
