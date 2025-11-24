import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import warnings
from absl import logging

print("\n=== CSC580 Neural Network Sales Prediction ===\n")
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


# Load data
print("[INFO] Loading training and test data ...")
training_data_df = pd.read_csv("sales_data_training.csv")
test_data_df = pd.read_csv("sales_data_test.csv")

# Scale data
print("[INFO] Applying MinMax scaling...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)
scale_factor = scaler.scale_[8]
scale_min = scaler.min_[8]
print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scale_factor, scale_min))
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Split inputs and outputs
print("[INFO] Splitting Features and Labels...")
X_train = scaled_training_df.drop('total_earnings', axis=1).values
Y_train = scaled_training_df[['total_earnings']].values
X_test = scaled_testing_df.drop('total_earnings', axis=1).values
Y_test = scaled_testing_df[['total_earnings']].values

# Build model
print("[INFO] Building and training neural network model...")
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
print("[INFO] Training model for 50 epochs...")
history = model.fit(
    X_train, Y_train,
    epochs=50,
    shuffle=True,
    verbose=2,
    validation_split=0.2
)

# Evaluate model
print("[INFO] Evaluating model on test dataset...")
test_error_rate = model.evaluate(X_test, Y_test, verbose=0)

# Save model
model.save("trained_model.h5")
print("[INFO] Saving trained model to disk (trained_model.h5)...")

# Load model
print("[INFO] Loading trained model back from disk...")
model = load_model("trained_model.h5")

# Load new product data
new_product_raw = pd.read_csv("proposed_new_product.csv", header=None)

# Assign correct feature names
feature_columns = training_data_df.drop('total_earnings', axis=1).columns
new_product_raw.columns = feature_columns

# Add dummy output column because scaler expects ALL columns
new_product_with_dummy = new_product_raw.copy()
new_product_with_dummy['total_earnings'] = 0.0
new_product_with_dummy = new_product_with_dummy[training_data_df.columns]

# Apply the same scaler
print("[INFO] Scaling new product row using same scaler...")
new_product_scaled_full = scaler.transform(new_product_with_dummy)

# Keep only the 9 input columns
new_product_scaled = new_product_scaled_full[:, :-1]

# Predict
print("[INFO] Generating prediction...")
prediction_scaled = model.predict(new_product_scaled)[0][0]

# Rescale prediction back to dollars
prediction = (prediction_scaled - scale_min) / scale_factor

# Print results
#print(f"\nThe mean squared error (MSE) for the test data set is: {test_error_rate}")
#print("Earnings Prediction for Proposed Product - ${:.2f}".format(prediction))

#Plot the curves 
print("[INFO] Plotting training and validation loss curves...")
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color="#1f77b4")
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color="#ff7f0e")
plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

annotation_text = (
    f"Test MSE: {test_error_rate:.6f}\n"
    f"Predicted Earnings: ${prediction:,.2f}"
)

plt.gca().text(
    0.98, 0.98, annotation_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor='black',
        alpha=0.85
    )
)

plt.tight_layout()
plt.show()
print("\n[INFO] Process completed successfully.\n")