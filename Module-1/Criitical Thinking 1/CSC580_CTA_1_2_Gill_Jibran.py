from random import randint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 500

def run_cycle(num_vars, coefficients, test_input=None):
    equation_terms = []
    for i, c in enumerate(coefficients):
        if c % 1 == 0:
            c = int(c)
        equation_terms.append(f"{c}x{i+1}")
    equation_str = " + ".join(equation_terms)

    print("\n==========================================")
    print("The algebraic equation being modeled is:")
    print(f"x = {equation_str}")
    print("==========================================\n")

    train_input = []
    train_output = []
    for i in range(TRAIN_SET_COUNT):
        inputs = [randint(0, TRAIN_SET_LIMIT) for _ in range(num_vars)]
        output = sum(c * x for c, x in zip(coefficients, inputs))
        print(f'Training Dataset row {i + 1}: {inputs} = ', end = "")
        print(output)
        train_input.append(inputs)
        train_output.append(output)

    model = LinearRegression(fit_intercept=False)
    model.fit(train_input, train_output)

    y_train_pred = model.predict(train_input)
    r2 = model.score(train_input, train_output)

    print("Training info:")
    print(f"  Samples   : {TRAIN_SET_COUNT}")
    print(f"  Variables : {num_vars}")
    print(f"  R^2 score : {r2:.4f}\n")

    if test_input is None:
        print("Enter test values to evaluate the model:")
        test_input = []
        for i in range(num_vars):
            val = float(input(f"Value for x{i+1}: "))
            test_input.append(val)
    else:
        print("Using predefined test values:")
        for i, v in enumerate(test_input, start=1):
            print(f"  x{i} = {v}")
        print()

    predicted = model.predict([test_input])[0]
    actual = sum(c * x for c, x in zip(coefficients, test_input))

    print("\n=========== RESULT ===========")
    print(f"Predicted Value : {predicted:.4f}")
    print(f"Actual Value    : {actual:.4f}")
    print("\nModel Learned Coefficients:")
    print(model.coef_)
    print("================================\n")

    train_actual = train_output
    train_pred = list(y_train_pred)
    min_y = min(train_actual + [actual, predicted])
    max_y = max(train_actual + [actual, predicted])

    plt.figure(figsize=(8, 6))
    plt.scatter(train_actual, train_pred, alpha=0.6, label="Training data")
    plt.scatter([actual], [predicted], marker="x", s=120, label="Test point")
    plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", label="Ideal y = x")
    plt.xlabel("Actual y")
    plt.ylabel("Predicted y")
    plt.title("Linear Regression: Actual vs Predicted")

    coeff_str = ", ".join(f"{c:.2f}" for c in model.coef_)
    text_block = (
        "Model Coeffs:\n[" + coeff_str + "]\n\n"
        + f"Actual y    = {actual:.4f}\n"
        + f"Predicted y = {predicted:.4f}"
    )

    ax = plt.gca()
    ax.text(
        0.02,
        0.98,
        text_block,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def manual_mode():
    num_vars = int(input("Enter number of variables (4-8): "))
    if num_vars < 4 or num_vars > 8:
        raise ValueError("Invalid number of variables. Must be between 4 and 8.")

    coefficients = []
    print("\nEnter the coefficient for each variable:")
    for i in range(num_vars):
        coef = float(input(f"Coefficient for x{i+1}: "))
        coefficients.append(coef)

    run_cycle(num_vars, coefficients)

def predefined_mode():
    presets = {
        "1": {
            "label": "4 variables",
            "num_vars": 4,
            "coeffs": [1, 2, 3, 4],
            "test": [10, 20, 30, 40],
        },
        "2": {
            "label": "5 variables",
            "num_vars": 5,
            "coeffs": [2, 4, 6, 8, 10],
            "test": [1, 2, 3, 4, 5],
        },
        "3": {
            "label": "6 variables",
            "num_vars": 6,
            "coeffs": [1, 0, -1, 2, -2, 3],
            "test": [5, 4, 3, 2, 1, 0],
        },
        "4": {
            "label": "7 variables",
            "num_vars": 7,
            "coeffs": [1, 1, 1, 1, 1, 1, 1],
            "test": [7, 6, 5, 4, 3, 2, 1],
        },
        "5": {
            "label": "8 variables",
            "num_vars": 8,
            "coeffs": [1, 2, 3, 4, 5, 6, 7, 8],
            "test": [8, 7, 6, 5, 4, 3, 2, 1],
        },
    }

    print("\nPredefined test cycles:")
    for key, cfg in presets.items():
        print(f"{key}. {cfg['label']} | coeffs={cfg['coeffs']} | test={cfg['test']}")

    choice = input("Select an option (1-5) or 'b' to go back: ").strip()
    if choice.lower() == "b":
        return
    if choice not in presets:
        print("Invalid choice.\n")
        return

    cfg = presets[choice]
    print(f"\nSelected: {cfg['label']}")
    print(f"Coefficients: {cfg['coeffs']}")
    print(f"Test values:  {cfg['test']}\n")

    run_cycle(cfg["num_vars"], cfg["coeffs"], cfg["test"])


def main():
    while True:
        print("========== MENU ==========")
        print("1. Enter manual values")
        print("2. Use a predefined test cycle")
        print("q. Quit")
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            manual_mode()
        elif choice == "2":
            predefined_mode()
        elif choice == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice.\n")
if __name__ == "__main__":
    main()