# Gait Recognition Project Report

I developed and evaluated gait-based user-identification workflows on three
smartphone sensor datasets. The notebooks cover data preparation, feature or
window extraction, model training, and evaluation.

## 1. Combined RightPocket

Notebook: [`gait_windowed_cnn.ipynb`](./Combined_RightPocket/gait_windowed_cnn.ipynb)

- Loaded 5,256,000 sensor rows from two sessions, covering 76 subjects; 36
  subjects had data in both sessions.
- Created 52,333 overlapping windows, each containing 200 samples across the
  three sensor axes.
- Implemented a session-based evaluation workflow with handcrafted-feature
  classical baselines and deep models, including a 1D CNN and CNN-BiLSTM.
- **Result:** The notebook's saved outputs do not include final model scores,
  so a verified accuracy cannot be reported here.

## 2. Koya Gait Combined

Notebook: [`koya_fast.ipynb`](./Koya_Gait_Combined_12Feature_Output/koya_fast.ipynb)

- Built a cycle-level, 390-class identification pipeline using 12 sensor
  features: accelerometer, gyroscope, and magnetometer axes and magnitudes.
- Used a session-based split with 29,363 training, 4,165 validation, and 8,327
  test cycles.
- Compared classical machine-learning models with CNN, LSTM, GRU, CNN-LSTM,
  and CNN-GRU models.
- **Best result:** CNN reached **89.46% test accuracy**, **0.8856 macro-F1**,
  and **96.33% top-5 accuracy**. The best classical baseline, SVM with PCA,
  reached **85.02% test accuracy** and **0.8426 macro-F1**.

## 3. Walking Recognition Dataset (77 people)

Notebook: [`code.ipynb`](./walking-recognition-dataset-77people/code.ipynb)

- Prepared smartphone inertial signals with chronological ordering, gap
  handling, low-pass filtering, and sliding-window extraction.
- Extracted 20,576 three-axis windows and 89 window features for **75 user
  classes**.
- Compared classical classifiers, a 1D CNN, and a CNN-BiLSTM.
- **Results:** XGBoost achieved **71.45% window accuracy** (macro-F1 **0.6992**).
  The 1D CNN achieved **51.92%**, and CNN-BiLSTM achieved **19.31%**. The
  notebook's CNN+XGBoost ensemble with 10-window voting reached **80.03%**.

## Evaluation note

The Koya results use a session-based test split. The walking-dataset notebook
uses a stratified split of windows, so windows from the same recording may
appear in both training and test sets; its scores should not be interpreted
as a strict unseen-session result. The Combined RightPocket notebook should be
rerun through its final evaluation cells to save and report its model scores.

Raw recordings, derived datasets, saved notebook outputs, and generated model
artifacts are not included in this repository.
