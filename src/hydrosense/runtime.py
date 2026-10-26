"""Separate bundled read-only resources from persistent desktop data."""
import os
from pathlib import Path
import sys


def resource_root():
    return Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]


def data_directory():
    if getattr(sys, 'frozen', False):
        return Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'HydroSense'
    return resource_root() / 'artifacts/hydro'
