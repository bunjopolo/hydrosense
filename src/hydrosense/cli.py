"""Batch training and scoring, including files larger than browser upload limits."""
import argparse
import json
from pathlib import Path
import joblib

from .data import read_file
from .model import Monitor, all_tags, train


def main():
    parser = argparse.ArgumentParser(description='HydroSense CSV/Excel training and replay')
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--data', required=True)
    training.add_argument('--sheet', default='')
    training.add_argument('--config', required=True)
    training.add_argument('--output', required=True)
    training.add_argument('--healthy-confirmed', action='store_true', required=True)
    scoring = sub.add_parser('score')
    scoring.add_argument('--data', required=True)
    scoring.add_argument('--sheet', default='')
    scoring.add_argument('--model', required=True, help='Only load a trusted local model you trained')
    scoring.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        config = json.loads(Path(args.config).read_text())
        bundle = train(read_file(args.data, config, args.sheet), config)
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


if __name__ == '__main__':
    main()
