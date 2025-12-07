###############################################
# CSC580 – Module 4 – Auto MPG Regression
# Option #2: Predicting Fuel Efficiency Using TensorFlow
# Jibran Gill
###############################################

from __future__ import absolute_import, division, print_function, unicode_literals

# ================
# Imports
# ================
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print("TensorFlow version:", tf.__version__)

np.random.seed(101)
tf.random.set_seed(101)

# ============================
# Step 1: Download the dataset
# ============================
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
print("Dataset path:", dataset_path)

# ============================
# Step 2: Load with pandas
# ============================
column_names = [
    'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
    'Acceleration', 'Model Year', 'Origin'
]

raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True
)

dataset = raw_dataset.copy()
print("\n=== Tail of raw dataset (for screenshot) ===")
print(dataset.tail())   # screenshot 1

# Drop rows with missing values
dataset = dataset.dropna()

# =======================
# Step 4: Train / test split
# =======================
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# =======================
# Step 5: Pairplot
# =======================
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
    diag_kind="kde"
)
plt.suptitle("Pairplot of Training Data", y=1.02)
plt.tight_layout()
plt.show()              # screenshot 2

# =======================
# Step 7–8: Stats (no MPG)
# =======================
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

print("\n=== Train statistics (for screenshot) ===")
print(train_stats.tail())   # screenshot 3

# =========================
# Step 9–11: Labels + norm
# =========================
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# =======================
# Step 12: Build model
# =======================
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )
    return model

model = build_model()

print("\n=== Model summary (for screenshot) ===")
model.summary()         # screenshot 4

# =======================
# Step 15: Try predictions (untrained)
# =======================
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print("\nExample predictions on 10 rows (untrained):")
print(example_result)

# =========================================
# Step 17–18: Train with EarlyStopping
# =========================================
EPOCHS = 1000

# Early stopping callback: stop if val_loss does not improve for 10 epochs
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\n=== Training model with EarlyStopping ===")
early_history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, tfdocs.modeling.EpochDots()]
)

print("\nTraining complete. Epochs actually run:", len(early_history.history['loss']))

# =========================================
# Step 19–21: Plot training history (MAE/MSE)
# =========================================
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# MAE plot with early stopping
plt.figure()
plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.title("Training & Validation MAE with Early Stopping")
plt.show()              # screenshot 5 (MAE plot)

# MSE plot (optional but nice)
plt.figure()
plotter.plot({'Early Stopping': early_history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.title("Training & Validation MSE with Early Stopping")
plt.show()

# You can inspect the last validation MAE as the “average error”
final_val_mae = early_history.history['val_mae'][-1]
print(f"\nFinal validation MAE after EarlyStopping: {final_val_mae:0.2f} MPG")

# =========================================
# Analyze generalization: evaluate on TEST set
# =========================================
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("\nTesting set Mean Abs Error: {:5.2f} MPG".format(mae))   # screenshot 6

# =========================================
# Make predictions on test set
# =========================================
test_predictions = model.predict(normed_test_data).flatten()

# Scatter: True vs Predicted
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.title("True vs Predicted MPG")
plt.show()              # screenshot 7

# =========================================
# Error distribution (normality)
# =========================================
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.show()              # screenshot 8