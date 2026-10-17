"""30-day SimpleRNN, predicting next-day temperature. Run from this folder."""
from pathlib import Path
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)
LOOKBACK = 30

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='raise')
    df = df.sort_values('date').reset_index(drop=True)
    if df['date'].duplicated().any() or df[['date', 'temperature_c']].isna().any().any():
        raise ValueError('Dates must be unique and values complete.')
    if not np.isfinite(df['temperature_c']).all():
        raise ValueError('Temperature must be finite.')
    if not df['date'].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Expected uninterrupted daily records.')
    if len(df) < 200:
        raise ValueError('At least 200 daily records required.')
    return df

def main():
    df = load_data(ROOT / 'temperature_forecasting_dataset.csv')
    values = df[['temperature_c']].to_numpy(dtype=np.float32)
    train_end, val_end = int(len(df)*0.70), int(len(df)*0.85)
    # Fit on training observations only, preventing scaler leakage.
    scaler = MinMaxScaler().fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)
    target_indices = np.arange(LOOKBACK, len(df))
    X = np.stack([scaled[i-LOOKBACK:i] for i in target_indices])
    y = scaled[target_indices]
    train = target_indices < train_end
    val = (target_indices >= train_end) & (target_indices < val_end)
    test = target_indices >= val_end
    # Earlier observed temperatures are valid context for rolling one-day forecasts.
    print('Train/validation/test samples:', train.sum(), val.sum(), test.sum())
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, 1)),
        tf.keras.layers.SimpleRNN(32, activation='tanh'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    model.summary()
    history = model.fit(X[train], y[train], validation_data=(X[val], y[val]),
                        epochs=100, batch_size=32, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=10, restore_best_weights=True)])
    predictions = scaler.inverse_transform(model.predict(X[test], verbose=0)).ravel()
    actual = values[target_indices[test], 0]
    persistence = values[target_indices[test]-1, 0]
    def metrics(pred):
        return {'MAE_C': float(mean_absolute_error(actual, pred)),
                'RMSE_C': float(np.sqrt(mean_squared_error(actual, pred))),
                'R2': float(r2_score(actual, pred))}
    scores = {'RNN': metrics(predictions), 'Yesterday_baseline': metrics(persistence)}
    print(json.dumps(scores, indent=2))
    (OUT / 'metrics.json').write_text(json.dumps(scores, indent=2))
    pd.DataFrame({'date': df.loc[target_indices[test], 'date'].to_numpy(),
                  'actual_c': actual, 'rnn_c': predictions,
                  'yesterday_baseline_c': persistence}).to_csv(OUT / 'test_predictions.csv', index=False)
    model.save(OUT / 'temperature_rnn.keras')
    joblib.dump(scaler, OUT / 'scaler.joblib')
    next_scaled = model.predict(scaled[-LOOKBACK:][None, ...], verbose=0)
    next_temp = float(scaler.inverse_transform(next_scaled)[0, 0])
    next_day = df['date'].iloc[-1] + pd.Timedelta(days=1)
    # Forecast follows the dataset's last date, not today's date.
    pd.DataFrame({'date': [next_day], 'forecast_c': [next_temp]}).to_csv(OUT / 'next_day_forecast.csv', index=False)
    print(f'Forecast for {next_day.date()}: {next_temp:.2f} C')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.loc[target_indices[test], 'date'], actual, label='Actual')
    ax.plot(df.loc[target_indices[test], 'date'], predictions, label='RNN')
    ax.set(ylabel='Temperature (C)', title='Test set: rolling next-day forecasts')
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / 'test_forecast.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Scaled MSE', title='Training loss')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / 'training_loss.png', dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    main()
