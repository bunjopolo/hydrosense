from __future__ import annotations
import base64
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import openpyxl

from hydrosense.data import read_upload
from hydrosense.demo import generate
from hydrosense.model import Monitor, all_tags, evaluate, read_csv, train, validate_config
from hydrosense.pi import PIClient
from hydrosense.service import State

ROOT = Path(__file__).resolve().parents[1]


class HydroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/hydro/assets.json').read_text())
        cls.frame = generate(1500)
        cls.bundle = train(cls.frame, cls.config)

    def values(self):
        return {t: float(self.frame.iloc[100][t]) for t in all_tags(self.config)}

    def test_arbitrary_names_and_variable_target_count(self):
        config = copy.deepcopy(self.config)
        rename = {t: f'Any tag / {i} [units]' for i, t in enumerate(all_tags(config))}
        for a in config['assets']:
            a['inputs'] = [rename[t] for t in a['inputs']]
            a['targets'] = [rename[t] for t in a['targets']]
            if a.get('running'):
                a['running']['tag'] = rename[a['running']['tag']]
        bundle = train(self.frame.rename(columns=rename), config)
        values = {rename[t]: v for t, v in self.values().items()}
        scored = evaluate(bundle, values)
        self.assertEqual([len(r['signals']) for r in scored], [2, 1, 2])
        self.assertEqual(scored[0]['signals'][0]['tag'], rename['U1.GB.TEMP'])

    def test_expected_value_changes_with_operating_inputs(self):
        values = self.values()
        values['U1.MW'] = 50
        first = evaluate(self.bundle, values)[0]['signals'][0]['expected']
        values['U1.MW'] = 80
        second = evaluate(self.bundle, values)[0]['signals'][0]['expected']
        self.assertGreater(second - first, 4)

    def test_retraining_changes_predictions_from_uploaded_values(self):
        frame = self.frame.copy()
        frame['U1.GB.TEMP'] += 20
        bundle = train(frame, self.config)
        old = evaluate(self.bundle, self.values())[0]['signals'][0]['expected']
        new = evaluate(bundle, self.values())[0]['signals'][0]['expected']
        self.assertAlmostEqual(new - old, 20, places=5)

    def test_target_not_used_as_own_predictor(self):
        values = self.values()
        before = evaluate(self.bundle, values)[0]['signals'][0]['expected']
        values['U1.GB.TEMP'] += 40
        after = evaluate(self.bundle, values)[0]
        self.assertEqual(before, after['signals'][0]['expected'])
        self.assertEqual(after['status'], 'deviation')
        config = copy.deepcopy(self.config)
        config['assets'][0]['inputs'].append('U1.GB.TEMP')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_config(config)

    def test_future_holdout_does_not_fit_model_or_calibration(self):
        frame = self.frame.copy()
        frame.loc[int(len(frame) * .8):, 'U1.GB.TEMP'] += 100
        changed = train(frame, self.config)
        np.testing.assert_array_equal(changed['models']['u1-bearing']['tolerance'], self.bundle['models']['u1-bearing']['tolerance'])
        self.assertEqual(evaluate(changed, self.values())[0]['signals'], evaluate(self.bundle, self.values())[0]['signals'])
        self.assertGreater(changed['reports'][0]['test_flag_rate'], .99)

    def test_integer_numeric_tags_are_supported(self):
        values = {t: np.int64(round(v)) for t, v in self.values().items()}
        self.assertNotEqual(evaluate(self.bundle, values)[0]['status'], 'data_quality')

    def test_missing_inputs_never_produce_healthy(self):
        values = self.values()
        del values['U1.MW']
        result = evaluate(self.bundle, values)[0]
        self.assertEqual(result['status'], 'data_quality')
        self.assertIsNone(result['score'])

    def test_domain_and_stopped_gate(self):
        values = self.values()
        values['U1.MW'] = 10000
        self.assertEqual(evaluate(self.bundle, values)[0]['status'], 'out_of_domain')
        values['U1.SPEED'] = 0
        self.assertEqual(evaluate(self.bundle, values)[0]['status'], 'not_running')

    def test_persistence_requires_fresh_observations(self):
        monitor = Monitor(self.bundle)
        values = self.values()
        values['U1.GB.TEMP'] += 40
        now = datetime.now(timezone.utc)
        times = dict.fromkeys(values, now)
        for _ in range(5):
            result = monitor.score(values, times, now)
            self.assertEqual(result[0]['consecutive'], 1)
            self.assertEqual(result[0]['status'], 'deviation')
        for i in (1, 2):
            stamp = now + timedelta(seconds=i * 30)
            result = monitor.score(values, dict.fromkeys(values, stamp), stamp)
        self.assertEqual(result[0]['status'], 'alert')

    def test_out_of_order_observation_is_not_scored_as_current(self):
        monitor = Monitor(self.bundle)
        values = self.values()
        now = datetime.now(timezone.utc)
        monitor.score(values, dict.fromkeys(values, now), now)
        earlier = now - timedelta(seconds=30)
        result = monitor.score(values, dict.fromkeys(values, earlier), now)[0]
        self.assertEqual(result['status'], 'data_quality')
        self.assertEqual(result['signals'], [])

    def test_stale_bad_quality_skew_and_future(self):
        now = datetime.now(timezone.utc)
        values = self.values()
        cases = [({'U1.MW': now - timedelta(minutes=10)}, None),
                 ({}, {'U1.MW': False}),
                 ({'U1.MW': now - timedelta(seconds=130)}, None),
                 ({'U1.MW': now + timedelta(minutes=1)}, None)]
        for override, bad in cases:
            times = {**dict.fromkeys(values, now), **override}
            quality = {**dict.fromkeys(values, True), **(bad or {})}
            result = Monitor(self.bundle).score(values, times, now, quality)[0]
            self.assertEqual(result['status'], 'data_quality')
            self.assertEqual(result['signals'], [])

    def test_serialization_and_column_order_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(self.bundle, path)
            loaded = joblib.load(path)
            reversed_values = dict(reversed(list(self.values().items())))
            self.assertEqual(evaluate(loaded, reversed_values), evaluate(self.bundle, self.values()))

    def test_csv_validation(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV column'):
            read_csv('timestamp,a,a\n2026-01-01T00:00:00Z,1,2', 'timestamp')
        with self.assertRaisesRegex(ValueError, 'duplicate timestamps'):
            read_csv('timestamp,a\n2026-01-01T00:00:00Z,1\n2026-01-01T00:00:00Z,2', 'timestamp')
        with self.assertRaisesRegex(ValueError, 'timezone'):
            read_csv('timestamp,a\n2026-01-01 00:00:00,1', 'timestamp')
        frame = read_csv('time,odd.tag name\n2026-01-01 00:00:00,1', 'time', 'America/Vancouver')
        self.assertEqual(frame['time'].iloc[0].hour, 8)

    def test_excel_named_sheet_preserves_headers_and_local_time(self):
        book = openpyxl.Workbook()
        book.active.title = 'Plant'
        book.active.append(['Time', 'bearing / temp #1'])
        book.active.append([datetime(2026, 1, 1), 45.2])
        stream = io.BytesIO()
        book.save(stream)
        payload = {'xlsx': base64.b64encode(stream.getvalue()).decode(), 'sheet': 'Plant'}
        frame = read_upload(payload, {'timestamp_column': 'Time', 'source_timezone': 'America/Vancouver'})
        self.assertEqual(list(frame.columns), ['Time', 'bearing / temp #1'])
        self.assertEqual(frame.Time.iloc[0].hour, 8)

    def test_training_rejects_insufficient_or_missing_tags(self):
        with self.assertRaisesRegex(ValueError, '250'):
            train(self.frame.iloc[:100], self.config)
        with self.assertRaisesRegex(ValueError, 'Missing configured tags'):
            train(self.frame.drop(columns=['U1.MW']), self.config)

    def test_bad_rows_excluded_not_imputed(self):
        frame = self.frame.copy()
        frame.loc[:9, 'U1.GB.TEMP'] = np.nan
        result = train(frame, self.config)
        self.assertEqual(result['reports'][0]['excluded_rows'], 10)


class PIAndServiceTests(unittest.TestCase):
    def test_pi_mapping_quality_and_partial_response(self):
        client = PIClient({'url': 'https://example.invalid/piwebapi', 'web_ids': {'odd tag': 'id1', 'bad': 'id2', 'missing': 'id3'}})
        client.get = Mock(return_value={'Items': [
            {'WebId': 'id1', 'Value': {'Value': 42, 'Good': True, 'Timestamp': '2026-01-01T00:00:00Z'}},
            {'WebId': 'id2', 'Value': {'Value': {'Name': 'No Data'}, 'Good': False, 'Timestamp': '2026-01-01T00:00:00Z'}}]})
        values, times, quality = client.snapshot(['odd tag', 'bad', 'missing'])
        self.assertEqual(values, {'odd tag': 42., 'bad': None, 'missing': None})
        self.assertEqual(quality, {'odd tag': True, 'bad': False, 'missing': False})
        self.assertIsNotNone(times['odd tag'].tzinfo)

    def test_pi_point_resolution_preserves_tag(self):
        client = PIClient({'url': 'https://example.invalid/piwebapi', 'data_server': 'archive'})
        client.get = Mock(return_value={'WebId': 'abc'})
        client.resolve(['U1 bearing / temperature'])
        client.get.assert_called_once_with('/points', {'path': '\\\\archive\\U1 bearing / temperature'})

    def test_tls_required_and_no_redirects(self):
        with self.assertRaises(ValueError):
            PIClient({'url': 'http://example.invalid'})
        with self.assertRaises(ValueError):
            PIClient({'url': 'https://user:secret@example.invalid'})
        client = PIClient({'url': 'https://example.invalid/piwebapi'})
        client.session.get = Mock(return_value=Mock(status_code=302))
        with self.assertRaisesRegex(ValueError, '302'):
            client.get('/system')
        self.assertFalse(client.session.get.call_args.kwargs['allow_redirects'])
        self.assertTrue(client.session.verify)

    def test_service_training_reload_replay_and_secret_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            state = State(directory)
            state.action('/connection', {'url': 'https://example.invalid/piwebapi', 'auth': 'basic', 'username': 'test', 'secret': 'DO-NOT-PERSIST'})
            self.assertNotIn('DO-NOT-PERSIST', (Path(directory) / 'connection.json').read_text())
            state.action('/demo', {})
            deadline = time.monotonic() + 20
            while state.job['status'] == 'training' and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertEqual(state.job['status'], 'complete', state.job)
            state.action('/replay', {'steps': 3})
            self.assertEqual(state.replay_index, 3)
            self.assertEqual(len(state.results), 3)
            with self.assertRaisesRegex(ValueError, 'healthy plant CSV'):
                state.action('/live/start', {})
            restored = State(directory)
            self.assertEqual(restored.secret, '')
            self.assertTrue(restored.snapshot()['model']['compatible'])
            changed = copy.deepcopy(restored.config)
            changed['assets'][0]['targets'].append('New target')
            restored.action('/config', changed)
            with self.assertRaisesRegex(ValueError, 'Train a model'):
                restored.action('/replay', {})


if __name__ == '__main__':
    unittest.main()
