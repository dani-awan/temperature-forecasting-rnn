from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RNN Temperature Forecast", layout="wide")

root = Path(__file__).resolve().parent
out = root / "outputs"

st.title("Temperature Forecasting with RNN")
st.caption("Presented by Muhammad Danial")
st.info("Simulated classroom data | Previous 30 days → next-day temperature")

required = ["metrics.json", "test_predictions.csv", "next_day_forecast.csv"]
missing = [name for name in required if not (out / name).exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.stop()

metrics = json.loads((out / "metrics.json").read_text())
predictions = pd.read_csv(out / "test_predictions.csv", parse_dates=["date"])
forecast = pd.read_csv(out / "next_day_forecast.csv", parse_dates=["date"])

a, b, c = st.columns(3)
a.metric("MAE", f"{metrics['RNN']['MAE_C']:.2f} °C")
b.metric("RMSE", f"{metrics['RNN']['RMSE_C']:.2f} °C")
c.metric("R²", f"{metrics['RNN']['R2']:.3f}")
st.caption("Lower MAE/RMSE is better. R² is not classification accuracy.")

graph_tab, results_tab, details_tab = st.tabs(
    ["Forecast & Graph", "Test Results", "Model Details"]
)

with graph_tab:
    st.subheader("Next-day Forecast")
    st.metric(
        forecast["date"].iloc[0].strftime("%d %B %Y"),
        f"{forecast['forecast_c'].iloc[0]:.2f} °C",
    )
    st.caption("This date follows the dataset's last observation; it is not live weather.")

    st.subheader("Actual vs Predicted")
    days = st.slider("Test days to display", 7, len(predictions), min(90, len(predictions)))
    chart = predictions.tail(days).set_index("date")[
        ["actual_c", "rnn_c", "yesterday_baseline_c"]
    ].rename(columns={
        "actual_c": "Actual",
        "rnn_c": "RNN",
        "yesterday_baseline_c": "Yesterday baseline",
    })
    st.line_chart(chart)

with results_tab:
    comparison = pd.DataFrame([
        {
            "Model": name,
            "MAE (°C)": values["MAE_C"],
            "RMSE (°C)": values["RMSE_C"],
            "R²": values["R2"],
        }
        for name, values in metrics.items()
    ])
    st.subheader("Model Comparison")
    st.dataframe(comparison, hide_index=True)
    st.subheader("Test Predictions")
    st.dataframe(predictions, hide_index=True)
    st.download_button(
        "Download predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        "test_predictions.csv",
        "text/csv",
    )

with details_tab:
    st.markdown("""
### Model Setup
- **Input:** Previous 30 daily temperature readings.
- **Architecture:** SimpleRNN (32) → Dense (16, ReLU) → Dense (1).
- **Optimizer:** Adam.
- **Loss:** Mean squared error.
- **Data split:** Chronological 70% train, 15% validation, 15% test.
- **Scaling:** MinMaxScaler fitted on training data only.
- **Early stopping:** Validation loss, restoring best weights.
- **Testing:** Rolling one-day forecasts using observed past temperatures.
""")
    st.warning("This dataset is simulated. Real-world weather performance has not been evaluated.")
    for filename in ["training_loss.png", "test_forecast.png"]:
        if (out / filename).exists():
            st.image(str(out / filename))

st.divider()
st.header("Select Date to Predict Temperature")

import numpy as np
import joblib
import tensorflow as tf

daily = pd.read_csv(
    root / "temperature_forecasting_dataset.csv",
    parse_dates=["date"],
).sort_values("date")

last_date = daily["date"].iloc[-1].date()
first_date = last_date + pd.Timedelta(days=1)
max_date = last_date + pd.Timedelta(days=7)

selected_date = st.date_input(
    "Prediction date",
    value=first_date,
    min_value=first_date,
    max_value=max_date,
)

st.caption(
    "Dataset 31 December 2022 tak hai. "
    "Agley 7 din ki experimental temperature prediction available hai."
)

if st.button("Predict Temperature", type="primary"):
    try:
        model = tf.keras.models.load_model(
            out / "temperature_rnn.keras", compile=False
        )
        scaler = joblib.load(out / "scaler.joblib")
        values = daily[["temperature_c"]].tail(30).to_numpy(
            dtype=np.float32
        )
        window = scaler.transform(values)
        days_ahead = (selected_date - last_date).days

        with st.spinner("Predicting..."):
            for _ in range(days_ahead):
                prediction = model.predict(
                    window.reshape(1, 30, 1), verbose=0
                )
                temperature = float(
                    scaler.inverse_transform(prediction)[0, 0]
                )
                window = np.concatenate(
                    [window[1:], prediction.reshape(1, 1)], axis=0
                )

        st.metric(
            f"Estimated temperature — {selected_date:%d %B %Y}",
            f"{temperature:.2f} °C",
        )
        st.caption(
            "Simulated data par trained RNN. "
            "Ek din se aagay predictions recursively banayi gayi hain "
            "aur unki accuracy evaluate nahi hui. Rain/cloud prediction nahi."
        )
    except Exception as error:
        st.error(str(error))
