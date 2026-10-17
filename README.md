TEMPERATURE FORECASTING WITH SimpleRNN

Windows / Anaconda Prompt:
  conda create -n temperature_rnn python=3.11 -y
  conda activate temperature_rnn
  cd "PATH_TO_EXTRACTED_FOLDER\Temperature_RNN"
  python -m pip install -r requirements.txt
  python train_rnn.py

Jupyter: open Temperature_RNN.ipynb in the same folder and run cells.
Google Colab: upload the notebook, CSV and train_rnn.py into /content.

Input: 1,826 daily observations, 2018-01-01 through 2022-12-31.
The data_type column identifies simulated classroom data, not real weather.
Input tensor: (samples, 30 days, 1 temperature feature).
Output: next-day temperature in degrees Celsius.
Chronological 70/15/15 observation split; no random time split.
Scaler fitted on training only. Early stopping uses validation only.
Test evaluation is rolling one-day prediction with observed past temperatures.
It is not a recursive multi-day forecast. No future observations enter inputs.
The final forecast targets 2023-01-01, the day after this dataset ends.
Compare RNN MAE/RMSE against yesterday's-temperature baseline (lower is better).
This package has not been trained in the creation environment (no TensorFlow).
Do not report invented accuracy: regression uses MAE, RMSE and R2.

Outputs after running: saved model, scaler, metrics, prediction CSVs and charts.
