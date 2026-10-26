from __future__ import annotations
import base64
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import openpyxl

from hydrosense.data import read_upload
from hydrosense.demo import generate
from hydrosense.model import MODEL_VERSION, Monitor, all_tags, config_for_frame, fingerprint, normalize_config, read_csv, train, validate_config
from hydrosense.pi import PIClient
from hydrosense.service import State

ROOT = Path(__file__).resolve().parents[1]


def fake_bundle(count=10):
    config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
    config.update(lookback=3, horizon=2, persistence=2)
    config['assets'] = [{'id': 'unit', 'name': 'Pilot unit', 'kind': 'Hydro unit', 'tags': [f'Any tag / {i} [unit]' for i in range(count)]}]
    return {'version': MODEL_VERSION, 'config': config, 'config_hash': fingerprint(config), 'sample_seconds': 30,
            'created_at': '2026-01-01T00:00:00Z', 'reports': [],
            'models': {'unit': {'mean': np.arange(count) + 10., 'tolerance': np.ones((2, count))}}}


class FixedPredictor:
    def __init__(self, stack):
        self.mean = stack['mean']
        self.calls = []

    def predict(self, values, times):
        self.calls.append(copy.deepcopy(values))
        q = np.stack([self.mean - .5, self.mean, self.mean + .5], axis=-1)
        future = pd.date_range(times[-1], periods=3, freq='30s')[1:]
        return np.stack([q, q]), future, {'isolation_forest': .4, 'lstm_autoencoder': .2}


class AllVariableMonitorTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fake_bundle()
        self.monitor = Monitor(self.bundle, FixedPredictor)
        self.values = dict(zip(all_tags(self.bundle['config']), range(10, 20)))
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def score(self, row, values=None, **kwargs):
        stamp = self.start + timedelta(seconds=row * 30)
        values = self.values if values is None else values
        return self.monitor.score(values, dict.fromkeys(self.values, stamp), stamp, live=False, **kwargs)[0]

    def warmup(self):
        for i in range(3):
            result = self.score(i)
        return result

    def test_all_ten_variables_always_have_signals_and_forecasts(self):
        result = self.warmup()
        self.assertEqual(len(result['signals']), 10)
        self.assertTrue(all(len(s['forecast']) == 2 for s in result['signals']))
        self.assertTrue(all(s['expected'] is None for s in result['signals']))
        result = self.score(3)
        self.assertTrue(all(s['expected'] is not None for s in result['signals']))

    def test_one_deviation_does_not_stop_other_nine_predictions(self):
        self.warmup()
        values = dict(self.values)
        tag = list(values)[4]
        values[tag] += 10
        first = self.score(3, values)
        second = self.score(4, values)
        self.assertEqual(first['signals'][4]['status'], 'deviation')
        self.assertEqual(second['signals'][4]['status'], 'alert')
        self.assertTrue(all(s['status'] == 'normal' for i, s in enumerate(second['signals']) if i != 4))
        self.assertTrue(all(len(s['forecast']) == 2 for s in second['signals']))

    def test_forecast_is_issued_before_observation_and_not_refit_to_actual(self):
        self.warmup()
        values = {t: v + 100 for t, v in self.values.items()}
        result = self.score(3, values)
        for signal in result['signals']:
            self.assertEqual(signal['expected'], self.values[signal['tag']])
            self.assertLess(pd.Timestamp(signal['forecast_issued_at']), pd.Timestamp(result['timestamp']))

    def test_repeated_snapshot_does_not_advance_persistence(self):
        self.warmup()
        values = {t: v + 10 for t, v in self.values.items()}
        for _ in range(5):
            result = self.score(3, values)
            self.assertTrue(all(s['consecutive'] == 1 for s in result['signals']))

    def test_quality_preserves_other_actual_values_but_resets_history(self):
        self.warmup()
        values = dict(self.values)
        values[list(values)[0]] = None
        result = self.score(3, values)
        self.assertEqual(result['status'], 'data_quality')
        self.assertEqual(len(result['signals']), 10)
        self.assertEqual(result['signals'][0]['status'], 'data_quality')
        self.assertTrue(all(s['actual'] is not None for s in result['signals'][1:]))
        self.assertTrue(all(not s['forecast'] for s in result['signals']))
        self.assertEqual(self.score(4)['status'], 'warming_up')

    def test_gap_and_out_of_order_do_not_compare_wrong_timestamps(self):
        self.warmup()
        result = self.score(10)
        self.assertEqual(result['status'], 'warming_up')
        self.assertTrue(all(s['expected'] is None for s in result['signals']))
        self.assertEqual(self.score(9)['status'], 'data_quality')

    def test_stale_and_future_pi_data(self):
        now = self.start + timedelta(minutes=10)
        result = self.monitor.score(self.values, dict.fromkeys(self.values, self.start), now)[0]
        self.assertEqual(result['status'], 'data_quality')
        result = self.monitor.score(self.values, dict.fromkeys(self.values, now + timedelta(minutes=2)), now)[0]
        self.assertEqual(result['status'], 'data_quality')

    def test_numeric_integer_tags(self):
        values = {t: np.int64(v) for t, v in self.values.items()}
        self.assertNotEqual(self.score(0, values)['status'], 'data_quality')

    def test_notification_and_pilot_export_identify_only_affected_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            state.bundle = self.bundle
            for i in range(3):
                result = self.score(i)
                state.record([result], result['timestamp'])
            values = dict(self.values)
            tag = list(values)[2]
            values[tag] += 10
            for i in (3, 4):
                result = self.score(i, values)
                state.record([result], result['timestamp'])
            self.assertEqual([e['tag'] for e in state.events], [tag, tag])
            self.assertEqual([e['status'] for e in state.events], ['deviation', 'alert'])
            rows = pd.read_csv(state.run_path)
            self.assertEqual(len(rows), 50)
            self.assertEqual(set(rows.tag), set(values))
            self.assertIn('forecast_issued_at', rows)
            result = self.score(5)
            state.record([result], result['timestamp'])
            self.assertEqual(state.events[-1]['status'], 'normal')


class ImportAndConfigurationTests(unittest.TestCase):
    def test_all_numeric_columns_are_discovered_without_mapping(self):
        frame = generate(300).rename(columns={'U1.MW': 'Any / name [MW]'})
        frame['Comment'] = 'not a sensor'
        config = config_for_frame(frame, fake_bundle()['config'], 'My unit')
        self.assertEqual(len(all_tags(config)), 10)
        self.assertIn('Any / name [MW]', all_tags(config))
        self.assertNotIn('Comment', all_tags(config))

    def test_old_inputs_are_now_also_forecast_targets(self):
        old = {'timestamp_column': 'timestamp', 'assets': [{'id': 'unit', 'inputs': ['Load'], 'targets': ['Temp']}]}
        self.assertEqual(all_tags(normalize_config(old)), ['Load', 'Temp'])
        self.assertEqual(validate_config(old)['assets'][0]['tags'], ['Load', 'Temp'])

    def test_csv_duplicate_and_timezone_validation(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV column'):
            read_csv('timestamp,a,a\n2026-01-01T00:00:00Z,1,2', 'timestamp')
        with self.assertRaisesRegex(ValueError, 'duplicate timestamps'):
            read_csv('timestamp,a\n2026-01-01T00:00:00Z,1\n2026-01-01T00:00:00Z,2', 'timestamp')
        with self.assertRaisesRegex(ValueError, 'timezone'):
            read_csv('timestamp,a\n2026-01-01 00:00:00,1', 'timestamp')
        self.assertEqual(read_csv('time,Tag\n2026-01-01 00:00:00,1', 'time', 'America/Vancouver').time.iloc[0].hour, 8)

    def test_excel_import_with_arbitrary_names(self):
        book = openpyxl.Workbook(); book.active.title = 'Plant'
        book.active.append(['Time', 'Bearing / A'])
        book.active.append([datetime(2026, 1, 1), 45.2])
        stream = io.BytesIO(); book.save(stream)
        frame = read_upload({'xlsx': base64.b64encode(stream.getvalue()).decode(), 'sheet': 'Plant'}, {'timestamp_column': 'Time', 'source_timezone': 'UTC'})
        self.assertEqual(list(frame.columns), ['Time', 'Bearing / A'])

    def test_service_import_configures_every_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            frame = generate(300).rename(columns={'timestamp': 'Time'})
            state.action('/csv', {'csv': frame.to_csv(index=False), 'timestamp_column': 'Time', 'auto_configure': True, 'purpose': 'training', 'unit_name': 'Unit X'})
            self.assertEqual(len(all_tags(state.config)), 10)
            self.assertEqual(state.config['assets'][0]['name'], 'Unit X')
            self.assertEqual(len(state.training), 300)

    def test_old_artifact_requires_retraining(self):
        with self.assertRaisesRegex(ValueError, 'Retrain'):
            Monitor({'version': 1})


class RealModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
        cls.config.update(lookback=8, horizon=3, epochs=1, hidden_size=8)
        cls.frame = generate(350).rename(columns={'U1.MW': 'A / load', 'U1.GB.TEMP': 'B.temp [C]'})
        cls.config['assets'][0]['tags'] = ['A / load', 'B.temp [C]']
        cls.progress_events = []
        cls.bundle = train(cls.frame, cls.config, progress_detail=cls.progress_events.append)

    def score_rows(self, bundle):
        monitor = Monitor(bundle)
        for _, row in self.frame.iloc[:10].iterrows():
            now = row.timestamp.to_pydatetime()
            values = {t: float(row[t]) for t in all_tags(self.config)}
            results = monitor.score(values, dict.fromkeys(values, now), now, live=False)
        return results

    def test_real_tft_forecasts_both_tags_and_lstm_and_iso_run(self):
        result = self.score_rows(self.bundle)[0]
        self.assertEqual(len(result['signals']), 2)
        self.assertTrue(all(len(s['forecast']) == 3 for s in result['signals']))
        self.assertTrue(all(s['expected'] is not None for s in result['signals']))
        self.assertEqual(set(result['components']), {'isolation_forest', 'lstm_autoencoder'})
        for signal in result['signals']:
            self.assertGreater(pd.Timestamp(signal['forecast'][0]['timestamp']), pd.Timestamp(result['timestamp']))

    def test_cpu_training_reports_batch_progress_without_premature_completion(self):
        events = self.progress_events
        percentages = [event['percent'] for event in events]
        self.assertEqual(percentages, sorted(percentages))
        self.assertEqual(percentages[0], 0)
        self.assertEqual(percentages[-1], 97)  # Service still needs to save and activate.
        for stage in ('TFT epoch 1/1', 'LSTM epoch 1/1'):
            updates = [event['percent'] for event in events if stage in event['message']]
            self.assertGreater(len(set(updates)), 1)

    def test_model_roundtrip_and_notebook_activation(self):
        from hydrosense.cli import activate
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'export.joblib'
            joblib.dump(self.bundle, model)
            loaded = joblib.load(model)
            self.assertEqual(self.score_rows(loaded), self.score_rows(self.bundle))
            target = Path(directory) / 'app'
            activate(model, target)
            state = State(target)
            self.assertTrue(state.snapshot()['model']['compatible'])
            self.assertEqual(state.source, 'notebook')

    def test_test_block_never_fits_transforms_or_thresholds(self):
        modified = self.frame.copy()
        modified.loc[int(len(modified) * .8):, 'B.temp [C]'] += 1000
        bundle = train(modified, self.config)
        key = self.config['assets'][0]['id']
        np.testing.assert_array_equal(bundle['models'][key]['mean'], self.bundle['models'][key]['mean'])
        np.testing.assert_allclose(bundle['models'][key]['tolerance'], self.bundle['models'][key]['tolerance'], rtol=1e-5)
        self.assertGreater(bundle['reports'][0]['tags'][1]['mae'], 900)

    def test_single_tag_model_and_predictions_respond_to_history(self):
        from hydrosense.forecasting import StackPredictor
        config = copy.deepcopy(self.config)
        config['assets'][0]['tags'] = ['A / load']
        bundle = train(self.frame, config)
        predictor = StackPredictor(bundle['models'][config['assets'][0]['id']])
        history = self.frame[['A / load']].iloc[:8].to_numpy()
        times = self.frame.timestamp.iloc[:8].tolist()
        original, _, _ = predictor.predict(history, times)
        changed, _, _ = predictor.predict(history + 20, times)
        self.assertEqual(original.shape, (3, 1, 3))
        self.assertTrue(np.isfinite(original).all())
        self.assertFalse(np.allclose(original, changed))


class TrainingSelectionTests(unittest.TestCase):
    def test_overlap_union_and_strict_boundaries(self):
        from hydrosense.selection import normalize_ranges
        self.assertEqual(normalize_ranges([[20, 50], [0, 30], [50, 60]], 100), [[0, 60]])
        for bad in ([[0, 101]], [[2, 2]], [[-1, 5]], [[False, 10]], [[1.5, 10]], None):
            with self.assertRaises(ValueError):
                normalize_ranges(bad, 100)

    def test_preview_retains_spikes_missing_values_and_all_numeric_tags(self):
        from hydrosense.selection import preview
        frame = generate(10000)
        frame.loc[1234, 'U1.MW'] = 99999
        frame.loc[1235, 'U1.MW'] = np.nan
        tags = [c for c in frame if c != 'timestamp']
        result = preview(frame, 'timestamp', tags)
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result['traces']), 10)
        values = result['traces'][0]['points']
        self.assertIn([1234, 99999.], values)
        self.assertIn([1235, None], values)
        self.assertLessEqual(len(values), 1000)
        zoom = preview(frame, 'timestamp', tags, 1230, 1240)
        self.assertEqual(len(zoom['traces'][0]['points']), 10)

    def test_selection_is_bound_to_import_and_empty_never_means_all(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            payload = {'csv': generate(500).to_csv(index=False), 'purpose': 'training', 'auto_configure': True}
            state.action('/csv', payload)
            revision = state.training_revision
            state.action('/training/selection', {'revision': revision, 'ranges': [[0, 200], [300, 500]]})
            self.assertEqual(state.snapshot()['selected_training_rows'], 400)
            self.assertEqual(state.snapshot()['training_rows'], 500)
            state.action('/training/selection', {'revision': revision, 'ranges': []})
            with self.assertRaisesRegex(ValueError, '250'):
                state.action('/train', {'healthy_confirmed': True, 'revision': revision})
            state.action('/csv', payload)
            with self.assertRaisesRegex(ValueError, 'changed'):
                state.action('/training/selection', {'revision': revision, 'ranges': [[0, 500]]})
            with self.assertRaisesRegex(ValueError, 'changed'):
                state.action('/train', {'healthy_confirmed': True, 'revision': revision})
            self.assertEqual(state.snapshot()['selected_training_rows'], 500)

    def test_excluded_values_never_fit_and_windows_do_not_bridge_ranges(self):
        from hydrosense.selection import selected_frame
        frame = generate(500)
        frame.loc[200:299, 'U1.MW'] = 1e9
        config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
        config.update(lookback=8, horizon=3, epochs=1, hidden_size=8)
        config['assets'][0]['tags'] = ['U1.MW']
        chosen = selected_frame(frame, [[0, 200], [300, 500]], 'timestamp')
        bundle = train(chosen, config)
        stack = bundle['models'][config['assets'][0]['id']]
        self.assertAlmostEqual(stack['mean'][0], chosen.iloc[:240]['U1.MW'].mean())
        self.assertEqual(bundle['sample_seconds'], 30)
        self.assertEqual(bundle['reports'][0]['fit_windows'], 219)
        self.assertEqual(bundle['training_selection']['selected_rows'], 400)
        self.assertEqual(bundle['training_selection']['ranges'][1]['start_time'], frame.iloc[300].timestamp.isoformat())


class PITests(unittest.TestCase):
    def test_live_rejects_cadence_mismatch_before_connecting(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            state.bundle = fake_bundle()
            state.config = state.bundle['config']
            state.bundle['sample_seconds'] = 60
            state.source = 'notebook'
            with self.assertRaisesRegex(ValueError, 'sampling cadence'):
                state.action('/live/start', {})

    def test_mapping_quality_partial_response(self):
        client = PIClient({'url': 'https://example.invalid/piwebapi', 'web_ids': {'odd tag': 'id1', 'bad': 'id2', 'missing': 'id3'}})
        client.get = Mock(return_value={'Items': [
            {'WebId': 'id1', 'Value': {'Value': 42, 'Good': True, 'Timestamp': '2026-01-01T00:00:00Z'}},
            {'WebId': 'id2', 'Value': {'Value': {'Name': 'No Data'}, 'Good': False, 'Timestamp': '2026-01-01T00:00:00Z'}}]})
        values, _, quality = client.snapshot(['odd tag', 'bad', 'missing'])
        self.assertEqual(values, {'odd tag': 42., 'bad': None, 'missing': None})
        self.assertEqual(quality, {'odd tag': True, 'bad': False, 'missing': False})

    def test_point_resolution_preserves_names(self):
        client = PIClient({'url': 'https://example.invalid/piwebapi', 'data_server': 'archive'})
        client.get = Mock(return_value={'WebId': 'abc'})
        client.resolve(['U1 bearing / temperature'])
        client.get.assert_called_once_with('/points', {'path': '\\\\archive\\U1 bearing / temperature'})

    def test_tls_and_no_credential_redirects(self):
        with self.assertRaises(ValueError): PIClient({'url': 'http://example.invalid'})
        with self.assertRaises(ValueError): PIClient({'url': 'https://user:secret@example.invalid'})
        client = PIClient({'url': 'https://example.invalid/piwebapi'})
        client.session.get = Mock(return_value=Mock(status_code=302))
        with self.assertRaisesRegex(ValueError, '302'): client.get('/system')
        self.assertFalse(client.session.get.call_args.kwargs['allow_redirects'])

    def test_secrets_remain_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            state.action('/connection', {'url': 'https://example.invalid/piwebapi', 'auth': 'basic', 'username': 'test', 'secret': 'DO-NOT-PERSIST'})
            self.assertNotIn('DO-NOT-PERSIST', (Path(directory) / 'connection.json').read_text())
            self.assertEqual(State(directory).secret, '')


if __name__ == '__main__':
    unittest.main()
