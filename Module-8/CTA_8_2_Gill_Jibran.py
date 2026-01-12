"""
CSC580 Portfolio Project - Option #2
Encoder-Decoder LSTM for Sequence-to-Sequence Prediction using Keras/TensorFlow
Author: Jibran Gill
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense



#Reproducibility
def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


#Encoder-Decoder Model Definition using Keras
def define_models(n_input: int, n_output: int, n_units: int):
    """
    Define the encoder-decoder models in Keras.

    Returns:
        train_model: Used for training with [encoder_input, decoder_input] -> decoder_output
        inf_encoder: Used to encode a source sequence for inference
        inf_decoder: Used to decode the target sequence one step at a time for inference
    """
    #Training Encoder
    encoder_inputs = Input(shape=(None, n_input), name="encoder_inputs")
    encoder = LSTM(n_units, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    #Training Decoder
    decoder_inputs = Input(shape=(None, n_output), name="decoder_inputs")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="train_model")

    #Inference Encoder
    inf_encoder = Model(encoder_inputs, encoder_states, name="inference_encoder")

    #Inference Decoder
    decoder_state_input_h = Input(shape=(n_units,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(n_units,), name="decoder_state_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    inf_decoder_outputs, out_h, out_c = decoder_lstm(
        decoder_inputs,
        initial_state=decoder_states_inputs
    )
    inf_decoder_states = [out_h, out_c]
    inf_decoder_outputs = decoder_dense(inf_decoder_outputs)

    inf_decoder = Model(
        [decoder_inputs] + decoder_states_inputs,
        [inf_decoder_outputs] + inf_decoder_states,
        name="inference_decoder"
    )

    return train_model, inf_encoder, inf_decoder

#Scalable Sequence-to-Sequence Toy Problem
def generate_sequence(length: int, n_unique: int):
    return [random.randint(1, n_unique - 1) for _ in range(length)]

def get_dataset(n_in: int, n_out: int, cardinality: int, n_samples: int):
    X1, X2, y = [], [], []
    for _ in range(n_samples):
        source = generate_sequence(n_in, cardinality)

        target = source[:n_out]
        target.reverse()

        target_in = [0] + target[:-1]

        src_encoded = to_categorical([source], num_classes=cardinality)
        tar_encoded = to_categorical([target], num_classes=cardinality)
        tar_in_encoded = to_categorical([target_in], num_classes=cardinality)

        X1.append(src_encoded)
        X2.append(tar_in_encoded)
        y.append(tar_encoded)

    return np.array(X1), np.array(X2), np.array(y)

def one_hot_decode(encoded_seq):
    return [int(np.argmax(vector)) for vector in encoded_seq]

#Inference: Predict a sequence step-by-step
def predict_sequence(infenc, infdec, source, n_steps: int, cardinality: int):
    state = infenc.predict(source, verbose=0)

    # Start-of-sequence token: one-hot for index 0
    target_seq = np.zeros((1, 1, cardinality), dtype=np.float32)
    target_seq[0, 0, 0] = 1.0

    output_probs = []

    for _ in range(n_steps):
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)

        output_probs.append(yhat[0, 0, :])

        state = [h, c]
        target_seq = yhat  # feed prediction back in

    return np.array(output_probs)

#Plotting helpers
def _metric_key(history_dict: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in history_dict:
            return k
    return None

def plot_training_history(history, filename="training_history.png"):
    h = history.history
    loss_key = _metric_key(h, ["loss"])
    acc_key = _metric_key(h, ["accuracy", "acc"])

    plt.figure()
    if loss_key:
        plt.plot(h[loss_key], label="Training Loss")
    if acc_key:
        plt.plot(h[acc_key], label="Training Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Encoder–Decoder Training Loss and Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_prediction_accuracy(correct: int, total: int, filename="prediction_accuracy.png"):
    incorrect = total - correct
    plt.figure()
    plt.bar(["Correct", "Incorrect"], [correct, incorrect])
    plt.ylabel("Count")
    plt.title(f"Sequence Prediction Accuracy ({total} Samples)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_decoder_confidence(predicted_probs: np.ndarray, filename="decoder_confidence.png"):
    confidences = predicted_probs.max(axis=1)
    plt.figure()
    plt.plot(range(1, len(confidences) + 1), confidences, marker="o")
    plt.xlabel("Decoder Time Step")
    plt.ylabel("Max Softmax Probability")
    plt.title("Decoder Confidence per Output Step")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

#Main runner: Train, Evaluate, Print samples
def main():
    set_seed(42)
    # Configure problem
    n_features = 50 + 1
    n_steps_in = 6
    n_steps_out = 3
    n_units = 128

    # Output artifacts (screenshots + text)
    out_txt = "output_predictions.txt"
    plot1 = "training_history.png"
    plot2 = "prediction_accuracy.png"
    plot3 = "decoder_confidence.png"

    # Quick sanity check: dataset shapes (matches assignment expectation)
    X1_check, X2_check, y_check = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    # Expect: (1, 1, 6, 51), (1, 1, 3, 51), (1, 1, 3, 51)
    print(X1_check.shape, X2_check.shape, y_check.shape)
    print("X1=%s, X2=%s, y=%s" % (
        one_hot_decode(X1_check[0][0]),
        one_hot_decode(X2_check[0][0]),
        one_hot_decode(y_check[0][0])
    ))

    #Training dataset
    n_train = 30000
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, n_train)

    # Keras expects (samples, timesteps, features); remove the extra singleton dimension
    X1 = X1.squeeze(axis=1)  # (n_train, 6, 51)
    X2 = X2.squeeze(axis=1)  # (n_train, 3, 51)
    y = y.squeeze(axis=1)    # (n_train, 3, 51)

    # Define and compile models
    train, infenc, infdec = define_models(n_features, n_features, n_units)
    train.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train
    history = train.fit(
        [X1, X2],
        y,
        epochs=20,        #epocs can be bumped if accuracy is low
        batch_size=64,
        verbose=0
    )

    # Plot training history
    plot_training_history(history, filename=plot1)

    # Evaluate on 100 fresh samples
    total, correct = 100, 0
    sample_lines = []

    # Store one example prediction probs for confidence plot
    example_pred_probs = None
    example_line = None

    for i in range(total):
        X1t, X2t, yt = get_dataset(n_steps_in, n_steps_out, n_features, 1)
        X1t = X1t.squeeze(axis=1)  # (1, 6, 51)
        yt = yt.squeeze(axis=1)    # (1, 3, 51)

        pred_probs = predict_sequence(infenc, infdec, X1t, n_steps_out, n_features)

        y_true = one_hot_decode(yt[0])
        y_hat = one_hot_decode(pred_probs)

        if np.array_equal(y_true, y_hat):
            correct += 1

        # Saving first 10 samples
        if i < 10:
            x_decoded = one_hot_decode(X1t[0])
            line = f"X={x_decoded} y={y_true}, yhat={y_hat}"
            sample_lines.append(line)

            # Confidence Plot
            if example_pred_probs is None:
                example_pred_probs = pred_probs
                example_line = line

    accuracy = (correct / total) * 100.0

    # Plot correct vs incorrect
    plot_prediction_accuracy(correct, total, filename=plot2)

    # Plot decoder confidence
    if example_pred_probs is not None:
        plot_decoder_confidence(example_pred_probs, filename=plot3)

    hist = history.history
    final_loss = hist.get("loss", [None])[-1]
    final_acc = hist.get("accuracy", hist.get("acc", [None]))[-1]

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("CSC580 Option #2 - Encoder–Decoder LSTM Seq2Seq\n")
        f.write("Toy Problem: target = reverse(first 3 items of 6-item input)\n\n")

        f.write(f"Training samples: {n_train}\n")
        f.write(f"n_features={n_features}, n_steps_in={n_steps_in}, n_steps_out={n_steps_out}, n_units={n_units}\n")
        f.write(f"Final training loss: {final_loss}\n")
        f.write(f"Final training accuracy: {final_acc}\n\n")

        f.write(f"Evaluation (n={total}): Accuracy: {accuracy:.2f}%\n\n")

        f.write("Sample predictions:\n")
        for line in sample_lines:
            f.write(line + "\n")

        f.write("\nSaved plots (for screenshots):\n")
        f.write(f"- {plot1}\n")
        f.write(f"- {plot2}\n")
        f.write(f"- {plot3} (confidence per decoder step)\n")

        if example_line:
            f.write("\nExample used for decoder confidence plot:\n")
            f.write(example_line + "\n")

    #Console output
    print(f"\nAccuracy: {accuracy:.2f}%\n")
    print("Sample predictions:")
    for line in sample_lines:
        print(line)

    print("\nSaved artifacts:")
    print(f"- {os.path.abspath(out_txt)}")
    print(f"- {os.path.abspath(plot1)}")
    print(f"- {os.path.abspath(plot2)}")
    print(f"- {os.path.abspath(plot3)}")

if __name__ == "__main__":
    main()