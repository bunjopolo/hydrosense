"""Explicitly synthetic hydro signals for end-to-end testing, not plant data."""
import numpy as np
import pandas as pd


def generate(rows=3000, seed=42, faults=False, start='2026-01-01'):
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    load = 65 + 22 * np.sin(t / 34) + rng.normal(0, 1, rows)
    cooling = 17 + 3 * np.sin(t / 67)
    ambient = 14 + 5 * np.sin(t / 53)
    mva = load * 1.1
    frame = pd.DataFrame({
        'timestamp': pd.date_range(start, periods=rows, freq='30s', tz='UTC'),
        'U1.MW': load, 'U1.SPEED': 150 + rng.normal(0, .1, rows), 'U1.COOLING.TEMP': cooling,
        'U1.GB.TEMP': 26 + .25 * load + .5 * cooling + rng.normal(0, .3, rows),
        'U1.GB.VIB': .4 + .009 * load + rng.normal(0, .025, rows),
        'U1.STATOR.TEMP': 30 + .6 * load + .5 * cooling + rng.normal(0, .5, rows),
        'T1.MVA': mva, 'SITE.AMBIENT': ambient,
        'T1.TOP.OIL.TEMP': 25 + .4 * mva + .6 * ambient + rng.normal(0, .4, rows),
        'T1.WINDING.TEMP': 35 + .6 * mva + .6 * ambient + rng.normal(0, .5, rows),
    })
    if faults:
        frame.loc[rows // 3:rows * 2 // 3, 'U1.GB.TEMP'] += np.linspace(0, 12, rows * 2 // 3 - rows // 3 + 1)
        frame.loc[rows * 3 // 4:, 'T1.TOP.OIL.TEMP'] += 10
    return frame
