"""Review imported history without changing its timestamps or joining separate runs."""
import numpy as np
import pandas as pd


def normalize_ranges(ranges, rows):
    if not isinstance(ranges, list) or len(ranges) > 1000:
        raise ValueError('Provide at most 1,000 included ranges.')
    valid = []
    for pair in ranges:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair):
            raise ValueError('Each range must contain two integer row boundaries.')
        start, end = pair
        if not 0 <= start < end <= rows:
            raise ValueError('Training ranges must be inside the imported file.')
        valid.append([start, end])
    merged = []
    for start, end in sorted(valid):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def selected_frame(frame, ranges, timestamp_column):
    ranges = normalize_ranges(ranges, len(frame))
    indices = np.concatenate([np.arange(start, end) for start, end in ranges]) if ranges else np.array([], dtype=int)
    selected = frame.iloc[indices].copy().reset_index(drop=True)
    cadence = float(frame[timestamp_column].diff().dt.total_seconds().median())
    selected.attrs['sample_seconds'] = cadence
    selected.attrs['training_selection'] = {
        'source_rows': len(frame), 'selected_rows': len(selected),
        'ranges': [{'start_row': start + 1, 'end_row': end,
                    'start_time': frame.iloc[start][timestamp_column].isoformat(),
                    'end_time': frame.iloc[end - 1][timestamp_column].isoformat()} for start, end in ranges],
    }
    return selected


def preview(frame, timestamp_column, tags, start=0, end=None):
    end = len(frame) if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(frame):
        raise ValueError('Choose a valid preview row range.')
    # Keep extrema, endpoints and missing-data markers per bucket. Spikes are
    # visible even when the full file is much larger than the screen.
    bounds = np.linspace(start, end, min(200, end - start) + 1, dtype=int)
    traces = []
    for tag in tags:
        values = pd.to_numeric(frame[tag], errors='coerce').to_numpy(dtype=float)
        points = []
        for left, right in zip(bounds[:-1], bounds[1:]):
            bucket = values[left:right]
            finite = np.flatnonzero(np.isfinite(bucket))
            offsets = {0, len(bucket) - 1}
            if len(finite):
                offsets.update([int(finite[np.argmin(bucket[finite])]), int(finite[np.argmax(bucket[finite])])])
            missing = np.flatnonzero(~np.isfinite(bucket))
            if len(missing):
                offsets.add(int(missing[0]))
            for offset in sorted(offsets):
                row = left + offset
                points.append([int(row), float(values[row]) if np.isfinite(values[row]) else None])
        traces.append({'tag': tag, 'points': points})
    return {'start': start, 'end': end, 'start_time': frame.iloc[start][timestamp_column].isoformat(),
            'end_time': frame.iloc[end - 1][timestamp_column].isoformat(), 'traces': traces,
            'reduced': end - start > 200}
