"""Tag-independent versions of the AirSense TFT, LSTM and Isolation Forest layers."""
from __future__ import annotations

import copy
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import IsolationForest


class SequenceAutoencoder(nn.Module):
    def __init__(self, count, hidden=16):
        super().__init__()
        self.encoder = nn.LSTM(count, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, count)

    def forward(self, values):
        _, (hidden, _) = self.encoder(values)
        repeated = hidden[-1].unsqueeze(1).expand(-1, values.shape[1], -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def neural_frame(values, timestamps):
    frame = pd.DataFrame(values, columns=[f'tag_{i}' for i in range(values.shape[1])])
    frame['step'] = np.arange(len(frame))
    frame['series'] = 'asset'
    minute = timestamps.hour * 60 + timestamps.minute + timestamps.second / 60
    frame['clock_sin'] = np.sin(2 * np.pi * minute / 1440)
    frame['clock_cos'] = np.cos(2 * np.pi * minute / 1440)
    return frame


def valid_dataset(dataset, bad):
    prefix = np.r_[0, np.cumsum(bad)]
    def keep(index):
        first = index.time_idx_first.to_numpy(dtype=int)
        last = index.time_idx_last.to_numpy(dtype=int)
        return prefix[last + 1] == prefix[first]
    try:
        return dataset.filter(keep)
    except ValueError:
        raise ValueError('No complete, regularly sampled windows. Clean the data or shorten history/horizon.') from None


@torch.no_grad()
def predict_dataset(model, dataset):
    predictions, actuals, previous = [], [], []
    model.eval()
    for x, y in dataset.to_dataloader(train=False, batch_size=128, num_workers=0):
        output = model(x).prediction
        multiple = isinstance(output, (list, tuple))
        predictions.append((torch.stack(list(output), dim=2) if multiple else output.unsqueeze(2)).cpu().numpy())
        actuals.append((torch.stack(list(y[0]), dim=2) if multiple else y[0].unsqueeze(2)).cpu().numpy())
        previous.append((torch.stack([t[:, -1] for t in x['encoder_target']], dim=1) if multiple else x['encoder_target'][:, -1:]).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals), np.concatenate(previous)


def fit_stack(raw, timestamps, settings, progress):
    progress('Preparing training windows', 0.)
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer
    from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

    torch.set_num_threads(min(4, torch.get_num_threads()))
    pl.seed_everything(42, workers=True, verbose=False)
    use_cuda = settings.get('device') in ('cuda', 'auto') and torch.cuda.is_available()
    if settings.get('device') == 'cuda' and not use_cuda:
        raise ValueError('CUDA GPU is unavailable. Select a GPU runtime in Colab or choose cpu.')
    device = torch.device('cuda' if use_cuda else 'cpu')
    count = raw.shape[1]
    lookback, horizon = settings['lookback'], settings['horizon']
    fit_end, cal_end = int(len(raw) * .6), int(len(raw) * .8)
    sample_seconds = settings['sample_seconds']
    complete = np.isfinite(raw).all(axis=1)
    gaps = np.r_[False, np.abs(np.diff(timestamps.asi8) / 1e9 - sample_seconds) > max(.001, sample_seconds * .05)]
    bad = ~complete | gaps
    fit_rows = raw[:fit_end][complete[:fit_end]]
    if len(fit_rows) < max(100, lookback + horizon) or len(raw) - cal_end < horizon:
        raise ValueError('Not enough complete rows for chronological fit/calibration/test splits.')
    mean, scale = fit_rows.mean(axis=0), fit_rows.std(axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(mean) * 1e-6, 1e-6))
    normalized = (raw - mean) / scale
    filled = np.where(np.isfinite(normalized), normalized, 0.)
    frame = neural_frame(filled, timestamps)
    aliases = [f'tag_{i}' for i in range(count)]
    dataset = TimeSeriesDataSet(
        frame.iloc[:fit_end], time_idx='step', target=aliases if count > 1 else aliases[0], group_ids=['series'],
        min_encoder_length=lookback, max_encoder_length=lookback,
        min_prediction_length=horizon, max_prediction_length=horizon,
        time_varying_known_reals=['clock_sin', 'clock_cos'],
        time_varying_unknown_reals=aliases,
        target_normalizer=MultiNormalizer([TorchNormalizer(method='identity') for _ in aliases]) if count > 1 else TorchNormalizer(method='identity'),
        scalers={name: None for name in ['clock_sin', 'clock_cos']},
        add_relative_time_idx=True, randomize_length=False,
    )
    training = valid_dataset(dataset, bad)
    parameters = dataset.get_parameters()
    calibration = valid_dataset(TimeSeriesDataSet.from_parameters(parameters, frame.iloc[:cal_end], min_prediction_idx=fit_end), bad)
    testing = valid_dataset(TimeSeriesDataSet.from_parameters(parameters, frame, min_prediction_idx=cal_end), bad)
    # Calibration/evaluation use non-overlapping forecast horizons for bounded work.
    calibration = calibration.filter(lambda idx: (idx.time_idx_first_prediction - fit_end) % horizon == 0)
    testing = testing.filter(lambda idx: (idx.time_idx_first_prediction - cal_end) % horizon == 0)
    if min(len(training), len(calibration), len(testing)) == 0:
        raise ValueError('Every split needs complete history plus future windows.')
    model = TemporalFusionTransformer.from_dataset(
        dataset, hidden_size=settings['hidden_size'], attention_head_size=2,
        hidden_continuous_size=8, dropout=.1, learning_rate=.003,
        loss=MultiLoss([QuantileLoss(quantiles=[.1, .5, .9]) for _ in aliases]) if count > 1 else QuantileLoss(quantiles=[.1, .5, .9]),
        output_size=[3] * count if count > 1 else 3, optimizer='adam', log_interval=-1,
    )
    class Progress(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            fraction = (trainer.current_epoch + (batch_idx + 1) / trainer.num_training_batches) / settings['epochs']
            progress(f'TFT epoch {trainer.current_epoch + 1}/{settings["epochs"]}', .05 + .60 * fraction)
    progress(f'Training TFT forecasts for all {count} tags', .05)
    trainer = pl.Trainer(max_epochs=settings['epochs'], accelerator='gpu' if use_cuda else 'cpu', devices=1,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, num_sanity_val_steps=0, gradient_clip_val=.1,
                         callbacks=[Progress()])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*does not have many workers.*')
        trainer.fit(model, train_dataloaders=training.to_dataloader(train=True, batch_size=settings['batch_size'], num_workers=0))
    model.eval().cpu()
    progress('Calibrating forecast residuals on later history', .65)
    cal_q, cal_actual, _ = predict_dataset(model, calibration)
    test_q, test_actual, previous = predict_dataset(model, testing)
    # All quantiles are normalized per tag. Sort only to correct crossing quantiles.
    cal_q, test_q = np.sort(cal_q, axis=-1), np.sort(test_q, axis=-1)
    error = np.abs(cal_actual - cal_q[..., 1])
    tolerance = np.maximum(np.quantile(error, .99, axis=0), .01) * scale[None, :]
    test_error = np.abs(test_actual - test_q[..., 1]) * scale[None, None, :]
    baseline = np.abs(test_actual - previous[:, None, :]) * scale[None, None, :]

    progress('Training Isolation Forest', .75)
    iso = IsolationForest(n_estimators=100, random_state=42, n_jobs=1).fit(filled[:fit_end][complete[:fit_end]])
    iso_reference = -iso.score_samples(filled[fit_end:cal_end][complete[fit_end:cal_end]])
    # Reconstruct past sequences, never future targets; do not cross quality gaps.
    starts = [i for i in range(lookback, fit_end + 1) if not bad[i - lookback:i].any()]
    ae = SequenceAutoencoder(count, settings['hidden_size']).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=.003)
    rng = np.random.default_rng(42)
    for epoch in range(settings['epochs']):
        ae.train()
        order = rng.permutation(starts)
        for offset in range(0, len(order), settings['batch_size']):
            batch = torch.tensor(np.stack([filled[i - lookback:i] for i in order[offset:offset + settings['batch_size']]]), dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = ((ae(batch) - batch) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.)
            optimizer.step()
            fraction = (epoch + min(offset + settings['batch_size'], len(order)) / len(order)) / settings['epochs']
            progress(f'LSTM epoch {epoch + 1}/{settings["epochs"]}', .80 + .15 * fraction)
    progress('Evaluating LSTM reconstruction', .95)
    ae.eval().cpu()
    cal_starts = [i for i in range(max(fit_end + lookback, lookback), cal_end + 1, lookback) if not bad[i - lookback:i].any()]
    if not cal_starts:
        raise ValueError('Calibration needs a complete LSTM history window.')
    with torch.no_grad():
        batch = torch.tensor(np.stack([filled[i - lookback:i] for i in cal_starts]), dtype=torch.float32)
        ae_reference = ((ae(batch) - batch) ** 2).mean(dim=(1, 2)).numpy()
    stack = {'parameters': parameters, 'tft_state': copy.deepcopy(model.state_dict()),
             'ae_state': copy.deepcopy(ae.state_dict()), 'iso': iso,
             'iso_limit': max(float(np.quantile(iso_reference, .99)), 1e-6),
             'ae_limit': max(float(np.quantile(ae_reference, .99)), 1e-6),
             'mean': mean, 'scale': scale, 'tolerance': tolerance, 'settings': settings,
             'tft_hparams': dict(hidden_size=settings['hidden_size'], attention_head_size=2,
                                 hidden_continuous_size=8, dropout=.1, learning_rate=.003)}
    report = {'fit_rows': fit_end, 'calibration_rows': cal_end - fit_end, 'test_rows': len(raw) - cal_end,
              'excluded_rows': int((~complete).sum()), 'fit_windows': len(training), 'test_windows': len(testing),
              'mae': test_error.mean(axis=(0, 1)).tolist(), 'baseline_mae': baseline.mean(axis=(0, 1)).tolist(),
              'tolerance': tolerance[0].tolist(), 'test_flag_rate': float((test_error[:, 0, :] >= tolerance[0]).any(axis=1).mean())}
    progress('Unit models trained and evaluated', 1.)
    return stack, report


class StackPredictor:
    def __init__(self, stack):
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
        self.stack = stack
        self.dataset_class = TimeSeriesDataSet
        count = len(stack['mean'])
        # from_dataset needs fitted dataset metadata; from_parameters recreates it from a harmless complete window.
        settings = stack['settings']
        times = pd.date_range('2000-01-01', periods=settings['lookback'] + settings['horizon'], freq=pd.Timedelta(seconds=settings['sample_seconds']))
        example = neural_frame(np.zeros((len(times), count)), times)
        dataset = TimeSeriesDataSet.from_parameters(stack['parameters'], example, predict=True)
        self.tft = TemporalFusionTransformer.from_dataset(dataset, **stack['tft_hparams'],
            loss=MultiLoss([QuantileLoss(quantiles=[.1, .5, .9]) for _ in range(count)]) if count > 1 else QuantileLoss(quantiles=[.1, .5, .9]),
            output_size=[3] * count if count > 1 else 3, optimizer='adam')
        self.tft.load_state_dict(stack['tft_state'])
        self.tft.eval()
        self.ae = SequenceAutoencoder(count, settings['hidden_size'])
        self.ae.load_state_dict(stack['ae_state'])
        self.ae.eval()

    @torch.no_grad()
    def predict(self, values, times):
        s = self.stack
        normalized = (np.asarray(values) - s['mean']) / s['scale']
        horizon = s['settings']['horizon']
        # Decoder unknown reals are placeholders only. TFT decoder sees clock features, not these values.
        future = np.repeat(normalized[-1:], horizon, axis=0)
        future_times = pd.date_range(times[-1], periods=horizon + 1, freq=pd.Timedelta(seconds=s['settings']['sample_seconds']))[1:]
        timestamps = pd.DatetimeIndex(list(times) + list(future_times))
        dataset = self.dataset_class.from_parameters(s['parameters'], neural_frame(np.concatenate([normalized, future]), timestamps), predict=True)
        x, _ = next(iter(dataset.to_dataloader(train=False, batch_size=1, num_workers=0)))
        output = self.tft(x).prediction
        q = (torch.stack(list(output), dim=2) if isinstance(output, (list, tuple)) else output.unsqueeze(2))[0].numpy()
        q = np.sort(q, axis=-1) * s['scale'][None, :, None] + s['mean'][None, :, None]
        batch = torch.tensor(normalized[None], dtype=torch.float32)
        ae_score = float(((self.ae(batch) - batch) ** 2).mean()) / s['ae_limit']
        iso_score = float(-s['iso'].score_samples(normalized[-1:])[0]) / s['iso_limit']
        return q, future_times, {'isolation_forest': iso_score, 'lstm_autoencoder': ae_score}
