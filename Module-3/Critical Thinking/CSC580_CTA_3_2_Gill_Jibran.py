###############################################
# CSC580 – Module 4 – Auto MPG Regression
# Option #2: Predicting Fuel Efficiency Using TensorFlow
# Jibran Gill
###############################################

from __future__ import absolute_import, division, print_function, unicode_literals


# STEP 0: Imports

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print("TensorFlow version:", tf.__version__)

# For reproducibility
np.random.seed(101)
tf.random.set_seed(101)


# STEP 1: Download the dataset
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
print("Dataset path:", dataset_path)

# STEP 2: Import dataset using Pandas
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

print("\n=== Tail of raw dataset (for screenshot - Step 3) ===")
print(dataset.tail())

# (STEP 3: take screenshot of dataset.tail() in your notebook/IDE)
# Basic cleanup: handle missing values

print("\nNumber of rows before dropping NA:", len(dataset))
dataset = dataset.dropna()
print("Number of rows after dropping NA:", len(dataset))

# STEP 4: Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print("\nTrain dataset shape:", train_dataset.shape)
print("Test dataset shape:", test_dataset.shape)

# STEP 5: Inspect the data (pairplot)

sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]],
    diag_kind="kde"
)

plt.suptitle("Pairplot of Training Data (MPG, Cylinders, Displacement, Weight)", y=1.02)
plt.tight_layout()
plt.show()


# STEP 7: Review statistics
train_stats = train_dataset.describe()
print("\n=== Full train_dataset.describe() ===")
print(train_stats)

#STEP 7/8: Remove MPG from stats (label), then transpose
train_stats.pop("MPG")
train_stats = train_stats.transpose()

print("\n=== Train statistics (without MPG) – for screenshot Step 8 ===")
print(train_stats.tail())

# (STEP 8: screenshot of tail of train_stats)

# STEP 9/10: Split features from labels

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

print("\nExample training features (first 5 rows):")
print(train_dataset.head())
print("\nExample training labels (first 5):")
print(train_labels.head())

# STEP 11: Normalize the data

def norm(x):
    """Normalize features using training data statistics."""
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print("\n=== Normalized training data (head) ===")
print(normed_train_data.head())

# STEP 12: Build the model

def build_model(loss="mse"):
    """
    Build a simple regression model with two hidden layers.
    By default uses MSE loss. For Step 22 comparison we can
    also build with 'mae' loss.
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss=loss,               # 'mse' or 'mae'
        optimizer=optimizer,
        metrics=['mae', 'mse']   # Track both for convenience
    )
    return model

# Main model (MSE loss as in the assignment)
model_mse = build_model(loss="mse")

# STEP 13: Inspect the model

print("\n=== Model summary (MSE-loss model) – for screenshot Step 14 ===")
model_mse.summary()

# (STEP 14: screenshot of the model summary)
# STEP 15: Try out the model on a small batch of examples

example_batch = normed_train_data[:10]
example_result = model_mse.predict(example_batch)

print("\n=== Example predictions on 10 training rows (untrained model) ===")
print(example_result)

# (STEP 16: if needed, capture another screenshot of summary/predictions)
# STEP 17–18: Train model

EPOCHS = 1000

print("\n=== Training MSE-loss model (this will take a bit) ===")
history_mse = model_mse.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

print("\nTraining complete for MSE-loss model.")

# STEP 19: Look at history object (training/validation stats)
hist_mse = pd.DataFrame(history_mse.history)
hist_mse['epoch'] = history_mse.epoch

print("\n=== Tail of training history for MSE-loss model – Step 20 screenshot ===")
print(hist_mse.tail())

# (STEP 20: screenshot of hist_mse.tail())
# STEP 21: Plot MAE and MSE over training

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# MAE plot
plt.figure()
plotter.plot({'MSE_loss_model': history_mse}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.title("MAE over Epochs – MSE-Loss Model")
plt.show()

# MSE plot
plt.figure()
plotter.plot({'MSE_loss_model': history_mse}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.title("MSE over Epochs – MSE-Loss Model")
plt.show()

# (STEP 21: screenshots of both plots)
# STEP 22: Compare two models: MSE-loss vs MAE-loss

# Build and train a second model that uses MAE as the loss
model_mae = build_model(loss="mae")

print("\n=== Training MAE-loss model (for comparison) ===")
history_mae = model_mae.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

print("\nTraining complete for MAE-loss model.")

# Evaluate both models on the *test* set
print("\n=== Evaluation on TEST set ===")

test_loss_mse, test_mae_mse, test_mse_mse = model_mse.evaluate(normed_test_data, test_labels, verbose=0)
print(f"MSE-loss model -> Test loss (MSE): {test_loss_mse:.3f}, Test MAE: {test_mae_mse:.3f}, Test MSE metric: {test_mse_mse:.3f}")

test_loss_mae, test_mae_mae, test_mse_mae = model_mae.evaluate(normed_test_data, test_labels, verbose=0)
print(f"MAE-loss model -> Test loss (MAE): {test_loss_mae:.3f}, Test MAE: {test_mae_mae:.3f}, Test MSE metric: {test_mse_mae:.3f}")

#side-by-side history plots

plt.figure()
plotter.plot(
    {
        'MSE_loss_model': history_mse,
        'MAE_loss_model': history_mae
    },
    metric="mae"
)
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.title("MAE comparison: MSE-loss vs MAE-loss models")
plt.show()
plt.figure()
plotter.plot(
    {
        'MSE_loss_model': history_mse,
        'MAE_loss_model': history_mae
    },
    metric="mse"
)
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.title("MSE comparison: MSE-loss vs MAE-loss models")
plt.show()

print("\n=== Interpretation hint for Step 22 (for your write-up) ===")
print("Compare the test MAE and MSE of both models. "
      "The model with lower MAE is generally better at predicting MPG on average. "
      "If both models have similar MAE/MSE, you can argue that both are 'useful', "
      "but pick the one with slightly better generalization (lower error on the test set).")