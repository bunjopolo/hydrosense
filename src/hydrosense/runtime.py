"""Paths for source resources and persistent local app data."""
from pathlib import Path


def resource_root():
    return Path(__file__).resolve().parents[2]


def data_directory():
    return resource_root() / 'artifacts/hydro'
