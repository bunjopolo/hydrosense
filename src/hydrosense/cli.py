"""Batch training and scoring, including files larger than browser upload limits."""
import argparse
import json
import shutil
import uuid
from pathlib import Path
import joblib

from .data import read_file
from .model import MODEL_VERSION, Monitor, all_tags, config_for_frame, fingerprint, train


def main():
    parser = argparse.ArgumentParser(description='HydroSense CSV/Excel training and replay')
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--data', required=True)
    training.add_argument('--sheet', default='')
    training.add_argument('--config', required=True)
    training.add_argument('--output', required=True)
    training.add_argument('--auto-tags', action='store_true', help='Forecast every numeric column as one unit')
    training.add_argument('--healthy-confirmed', action='store_true', required=True)
    scoring = sub.add_parser('score')
    scoring.add_argument('--data', required=True)
    scoring.add_argument('--sheet', default='')
    scoring.add_argument('--model', required=True, help='Only load a trusted local model you trained')
    scoring.add_argument('--output', required=True)
    activation = sub.add_parser('activate', help='Stop the service before activating a model you trained or trust')
    activation.add_argument('--model', required=True)
    activation.add_argument('--trust-model', action='store_true', required=True)
    args = parser.parse_args()
    if args.command == 'activate':
        print(activate(args.model))
    elif args.command == 'train':
        config = json.loads(Path(args.config).read_text())
        frame = read_file(args.data, config, args.sheet)
        if args.auto_tags:
            config = config_for_frame(frame, config)
        bundle = train(frame, config, print)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.output)
        print(json.dumps(bundle['reports'], indent=2))
    else:
        bundle = joblib.load(args.model)
        monitor = Monitor(bundle)
        frame = read_file(args.data, bundle['config'], args.sheet)
        import pandas as pd
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open('w') as target:
            for _, row in frame.iterrows():
                now = row[bundle['config']['timestamp_column']].to_pydatetime()
                values = {t: pd.to_numeric(row.get(t), errors='coerce') for t in all_tags(bundle['config'])}
                results = monitor.score(values, dict.fromkeys(values, now), now, live=False)
                target.write(json.dumps({'timestamp': now.isoformat(), 'assets': results}, allow_nan=False) + '\n')


def activate(path, directory=None):
    # joblib is executable serialization. Caller must explicitly trust this model.
    from .service import atomic_json
    from .runtime import data_directory
    bundle = joblib.load(path)
    if bundle.get('version') != MODEL_VERSION or bundle.get('config_hash') != fingerprint(bundle['config']):
        raise ValueError('Expected a matching all-tag HydroSense model. Retrain with this version.')
    Monitor(bundle)  # Verify all weights and fitted preprocessors before activation.
    directory = Path(directory or data_directory())
    directory.mkdir(parents=True, exist_ok=True)
    name = f'model-{uuid.uuid4().hex}.joblib'
    shutil.copyfile(path, directory / name)
    atomic_json(directory / 'config.json', bundle['config'])
    atomic_json(directory / 'active.json', {'file': name, 'source': 'notebook'})
    return 'Model activated. Restart HydroSense to load it.'


if __name__ == '__main__':
    main()
