'''
Gill_Jibran
Module-6 Option-2, Critical Thinking Assignment
CSC-580 Dr. Li
'''

import os
import glob
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

#Step 0: Loading Kagle Data
BASE_DIR = r"C:\Code\CSC580-Capstone Project\Module-6"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test1")

# Step 1: Load and preprocess with NumPy
IMG_SIZE = (94, 125)

def pixels_from_path(file_path):
    im = Image.open(file_path).convert("RGB")
    im = im.resize((IMG_SIZE[1], IMG_SIZE[0]))  
    np_im = np.array(im)                     
    return np_im

# Build "cats/*" and "dogs/*" equivalent lists from TRAIN_DIR
cat_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "cat.*.jpg")))
dog_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "dog.*.jpg")))

if len(cat_files) == 0 or len(dog_files) == 0:
    raise FileNotFoundError(
        "Could not find cat.*.jpg and dog.*.jpg in your train folder. "
        f"Checked: {TRAIN_DIR}"
    )

# Shape counting
shape_counts = defaultdict(int)
for i, cat in enumerate(cat_files[:1000]):
    if i % 100 == 0:
        print("shape scan index:", i)
    img_shape = pixels_from_path(cat).shape
    shape_counts[str(img_shape)] += 1

shape_items = list(shape_counts.items())
shape_items.sort(key=lambda x: x[1], reverse=True)

print("\nMost common shapes (top 10):")
for k, v in shape_items[:10]:
    print(k, "=>", v)

# Variables
validation_size = 0.1
img_size = IMG_SIZE
num_channels = 3
sample_size = 8192

print("\nSanity check one cat image shape:", pixels_from_path(cat_files[5]).shape)

# Sample sizes
SAMPLE_SIZE = 2048
valid_size = 512
print("\nloading training cat images...")
cat_train_set = np.asarray([pixels_from_path(p) for p in cat_files[:SAMPLE_SIZE]])
print("loading training dog images...")
dog_train_set = np.asarray([pixels_from_path(p) for p in dog_files[:SAMPLE_SIZE]])
print("loading validation cat images...")
cat_valid_set = np.asarray([pixels_from_path(p) for p in cat_files[-valid_size:]])
print("loading validation dog images...")
dog_valid_set = np.asarray([pixels_from_path(p) for p in dog_files[-valid_size:]])

x_train = np.concatenate([cat_train_set, dog_train_set], axis=0)
labels_train = np.asarray([1 for _ in range(SAMPLE_SIZE)] + [0 for _ in range(SAMPLE_SIZE)])
x_valid = np.concatenate([cat_valid_set, dog_valid_set], axis=0)
labels_valid = np.asarray([1 for _ in range(valid_size)] + [0 for _ in range(valid_size)])

labels_train = labels_train.reshape(-1, 1).astype("float32")
labels_valid = labels_valid.reshape(-1, 1).astype("float32")

x_train = x_train.astype("float32") / 255.0
x_valid = x_valid.astype("float32") / 255.0

print("\nDataset ready:")
print("x_train:", x_train.shape, "labels_train:", labels_train.shape)
print("x_valid:", x_valid.shape, "labels_valid:", labels_valid.shape)

# Step 2: Single hidden-layer model
# Adam optimizer
# 10 epochs
# shuffle
# MSE loss
print("\n====================")
print("STEP 2: single hidden-layer model")
print("====================")
total_pixels = img_size[0] * img_size[1] * 3
fc_size = 512
inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="ani_image")
x = layers.Flatten(name="flattened_img")(inputs)
x = layers.Dense(fc_size, activation="relu", name="first_layer")(x)
outputs = layers.Dense(1, activation="sigmoid", name="class")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
customAdam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=customAdam,
    loss="mean_squared_error",
    metrics=["binary_crossentropy", "mean_squared_error"],
)
print("# Fit model on training data")
history_step2 = model.fit(
    x_train,
    labels_train,
    batch_size=32,
    shuffle=True,   
    epochs=10,
    validation_data=(x_valid, labels_valid),
)

# Step 3: CNN with
# - one conv layer (24 kernels)
# - max pooling
# - two fully connected layers
# - Pearson correlation between preds and validation labels
print("\n====================")
print("STEP 3: conv model (24 kernels) + Pearson correlation")
print("====================")
fc_layer_size = 128
conv_inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="ani_image")
conv_layer = layers.Conv2D(24, kernel_size=3, activation="relu")(conv_inputs)
conv_layer = layers.MaxPool2D(pool_size=(2, 2))(conv_layer)
conv_x = layers.Flatten(name="flattened_features")(conv_layer)
conv_x = layers.Dense(fc_layer_size, activation="relu", name="first_layer")(conv_x)
conv_x = layers.Dense(fc_layer_size, activation="relu", name="second_layer")(conv_x)
conv_outputs = layers.Dense(1, activation="sigmoid", name="class")(conv_x)
conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)
customAdam = keras.optimizers.Adam(learning_rate=1e-6) 
conv_model.compile(
    optimizer=customAdam,
    loss="binary_crossentropy",
    metrics=["binary_crossentropy", "mean_squared_error"],
)
print("# Fit model on training data")
history_step3 = conv_model.fit(
    x_train,
    labels_train,
    batch_size=32,
    shuffle=True,
    epochs=5,
    validation_data=(x_valid, labels_valid),
)

preds = conv_model.predict(x_valid, verbose=0)
preds = np.asarray([p[0] for p in preds])
corr_step3 = np.corrcoef(preds, labels_valid.ravel())[0][1]
print("\nPearson correlation (Step 3):", corr_step3)

print("\n====================")
print("STEP 4: bigger conv model (add 48-kernel conv layer) + correlation + accuracy")
print("====================")

big_inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="ani_image")

x = layers.Conv2D(24, kernel_size=3, activation="relu")(big_inputs)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Conv2D(48, kernel_size=3, activation="relu")(x)
x = layers.MaxPool2D(pool_size=(2, 2))(x)

x = layers.Flatten(name="flattened_features")(x)
x = layers.Dense(fc_layer_size, activation="relu", name="first_layer")(x)
x = layers.Dense(fc_layer_size, activation="relu", name="second_layer")(x)
big_outputs = layers.Dense(1, activation="sigmoid", name="class")(x)

big_model = keras.Model(inputs=big_inputs, outputs=big_outputs)

big_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", "mean_squared_error"],
)

history_step4 = big_model.fit(
    x_train,
    labels_train,          # keep as (N,1) for Keras
    batch_size=32,
    shuffle=True,
    epochs=5,
    validation_data=(x_valid, labels_valid),
)

# ---- Make 1-D views for NumPy/Seaborn calculations ----
val_probs = big_model.predict(x_valid, verbose=0).reshape(-1)      # (N,)
val_labels_1d = labels_valid.reshape(-1)                           # (N,)

# Pearson correlation (must be 1-D)
corr_step4 = np.corrcoef(val_probs, val_labels_1d)[0][1]
print("\nPearson correlation (Step 4, 2 conv layers):", corr_step4)

# Correct validation accuracy @0.5 threshold (must be 1-D vs 1-D)
val_pred_labels = (val_probs >= 0.5).astype(int)
val_accuracy = (val_pred_labels == val_labels_1d.astype(int)).mean()
print("Validation accuracy @0.5 threshold:", val_accuracy)

# Scatterplot (Seaborn/Pandas require 1-D arrays)
print("val_probs shape:", val_probs.shape)
print("val_labels_1d shape:", val_labels_1d.shape)
print("labels_valid shape:", labels_valid.shape)
sns.scatterplot(x=val_probs, y=val_labels_1d)
plt.xlabel("Predicted probability (cat)")
plt.ylabel("True label (1=cat, 0=dog)")
plt.title("Step 4 Scatterplot: Predictions vs Validation Labels")
plt.show()

# Threshold analysis (fixed version of professor loop)
for i in range(1, 10):
    t = 0.1 * i
    mask = val_probs > t
    print(f"threshold: {t:.1f}")
    if mask.sum() == 0:
        print("  no samples above threshold")
    else:
        # Mean label among samples above threshold = fraction of cats in that slice
        print("  fraction cats above threshold:", val_labels_1d[mask].mean())

# Step 5: Utility functions to select an image + probability
print("\n====================")
print("STEP 5: utility functions (animal_pic + cat_index)")
print("====================")

def animal_pic(index):
    # convert back to uint8 for display
    img = (x_valid[index] * 255).astype(np.uint8)
    return Image.fromarray(img)

def cat_index(index):
    # IMPORTANT: use the passed index (the professor snippet had a bug using [124])
    return big_model.predict(np.asarray([x_valid[index]]), verbose=0)[0][0]

# Example
index = 600 if len(x_valid) > 600 else 0
print("probability of being a cat:", cat_index(index))
#animal_pic(index).show()


# Step 6: Save the model
print("\n====================")
print("STEP 6: saving model")
print("====================")

SAVE_PATH = os.path.join(BASE_DIR, "conv_model_big.keras")
big_model.save(SAVE_PATH)
print("Saved model to:", SAVE_PATH)

# Step 7: Simple interface: user picks index -> print prob + image
print("\n====================")
print("STEP 7: user interface (pick index)")
print("====================")

n = len(x_valid)
cats_end = valid_size  # 512

print(f"Valid index range: 0 to {n-1}")
print(f"Index meaning (based on how x_valid was built):")
print(f"  0 to {cats_end-1}   -> CAT validation images (label=1)")
print(f"  {cats_end} to {n-1} -> DOG validation images (label=0)")
print("Examples: try 0, 25, 400 (cats) or 600, 800, 1000 (dogs)")
print("Tip: type 'r' for a random index, or 'q' to quit.\n")

rng = np.random.default_rng()

while True:
    user_in = input("Enter an index (or 'r' random / 'q' quit): ").strip().lower()

    if user_in in ("q", "quit", "exit"):
        break

    if user_in in ("r", "rand", "random"):
        idx = int(rng.integers(0, n))
    else:
        try:
            idx = int(user_in)
        except ValueError:
            print("Please enter a valid integer, 'r', or 'q'.")
            continue

    if idx < 0 or idx >= n:
        print("Index out of range.")
        continue

    true_label = int(labels_valid.reshape(-1)[idx])  # 1=cat, 0=dog
    p = cat_index(idx)

    print(f"Index {idx} | true_label={true_label} (1=cat,0=dog) | predicted P(cat)={p:.6f}")
    animal_pic(idx).show()