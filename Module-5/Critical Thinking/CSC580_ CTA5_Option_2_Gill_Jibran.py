"""
CSC580 - Module 5 Option #2 Random Forest Classifier
Jibran Gill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#Load the data
np.random.seed(0)
df = pd.read_csv("iris_training.csv")

#Data processing - rename columns
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

print("\nDataset:")
print(df.head())

#Create training and test data using the 75/25 split
df["is_train"] = np.random.uniform(0, 1, len(df)) <= 0.75
print("\nSDataset with is_train column: ")
print(df.head())
train = df[df["is_train"]].copy()
test = df[~df["is_train"]].copy()
print("\n Train/Test counts: ")
print("Training samples:", len(train))
print("Test samples:", len(test))

#Preprocess the data
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train = train[features]
y_train = train["species"]
X_test = test[features]
y_test = test["species"]
print("\nFeature list: ")
print(features)
print("\n Encoded target values: ")
print(y_train.head(25).to_numpy())

#Training the Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    n_jobs=2
)
clf.fit(X_train, y_train)
print("\nRandom Forest trained: ")

#Train/Test accuracy for main model
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_preds_for_acc = clf.predict(X_test)
test_acc = accuracy_score(y_test, test_preds_for_acc)
print("\nMain Model Accuracy: ")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")

# Predicted probabilities (first 10 test samples)
probs = clf.predict_proba(X_test)
probs_df = pd.DataFrame(
    probs[:10],
    columns=["P(Setosa)", "P(Versicolor)", "P(Virginica)"]
)
print("\nPredicted probabilities (first 10): ")
print(probs_df)

#Add predicted class + confidence for first 10
preds_all = clf.predict(X_test)
conf_all = probs.max(axis=1) 
first10_summary = pd.DataFrame({
    "Actual Species": y_test.iloc[:10].to_numpy(),
    "Predicted Species": preds_all[:10],
    "Confidence (max prob)": conf_all[:10]
})
print("\nFirst 10: Actual vs Predicted + Confidence (optional screenshot):")
print(first10_summary)

#Confidence histogram
plt.figure(figsize=(7, 4))
plt.hist(conf_all, bins=10)
plt.title("Prediction Confidence Distribution (Test Set)")
plt.xlabel("Confidence (max predicted probability)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Predicted vs Actual (first 5 observations)
preds = clf.predict(X_test)
compare5 = pd.DataFrame({
    "Actual Species": y_test.iloc[:5].to_numpy(),
    "Predicted Species": preds[:5]
})
print("\nActual vs Predicted (first 5)")
print(compare5)

#Classification report
print("\nClassification Report (optional, good for write-up):")
print(classification_report(y_test, preds, digits=4))

#Show misclassified rows (if any)
mis_mask = (y_test.to_numpy() != preds)
mis_idx = test.index[mis_mask]

print("\nMisclassifications summary (optional, great for discussion):")
print(f"Total misclassified: {mis_mask.sum()} out of {len(test)}")

if mis_mask.sum() > 0:
    mis_df = test.loc[mis_idx, features + ["species"]].copy()
    mis_df["predicted"] = preds[mis_mask]
    print("\nMisclassified samples (showing up to first 10):")
    print(mis_df.head(10))
else:
    print("No misclassifications in this test split.")

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(
    cm,
    index=["Setosa", "Versicolor", "Virginica"],
    columns=["Setosa", "Versicolor", "Virginica"]
)
print("\nConfusion Matrix table: ")
print(cm_df)

#Confusion matrix normalized
cm_row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, cm_row_sums, out=np.zeros_like(cm, dtype=float), where=cm_row_sums != 0)

cm_norm_df = pd.DataFrame(
    cm_norm,
    index=["Setosa", "Versicolor", "Virginica"],
    columns=["Setosa", "Versicolor", "Virginica"]
)
print("\nNormalized Confusion Matrix (row-wise proportions):")
print(cm_norm_df.round(3))

# Heatmap-style plot
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix - Random Forest (Iris)")
plt.colorbar()
plt.xticks([0, 1, 2], ["Setosa", "Versicolor", "Virginica"])
plt.yticks([0, 1, 2], ["Setosa", "Versicolor", "Virginica"])

for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

#Feature importance
importances = pd.DataFrame({
    "Feature": features,
    "Importance": clf.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature importance scores: ")
print(importances)
plt.figure(figsize=(7, 4))
plt.bar(importances["Feature"], importances["Importance"])
plt.title("Feature Importances – Random Forest")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#Print top feature(s) explicitly
top_feature = importances.iloc[0]["Feature"]
top_importance = importances.iloc[0]["Importance"]
print("\nTop Feature:")
print(f"{top_feature} (importance = {top_importance:.4f})")

# Actual vs Predicted plot
plt.figure(figsize=(8, 4))
plt.scatter(range(len(y_test)), y_test, label="Actual", marker="o")
plt.scatter(range(len(preds)), preds, label="Predicted", marker="x")
plt.yticks([0, 1, 2], ["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Test Sample Index")
plt.ylabel("Species")
plt.title("Actual vs Predicted Species (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()

# Underfit vs Baseline
models = {
    "Underfit (max_depth=2)": RandomForestClassifier(max_depth=2, n_estimators=50, random_state=0),
    "Baseline": RandomForestClassifier(random_state=0),
    "More Complex (300 trees)": RandomForestClassifier(n_estimators=300, random_state=0),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    rows.append((name, train_acc, test_acc))
scores = pd.DataFrame(rows, columns=["Model", "Train Accuracy", "Test Accuracy"])

print("\nUnderfit vs Baseline vs More Complex")
print(scores)

#Plot train vs test accuracy comparison
plt.figure(figsize=(9, 4))
x = np.arange(len(scores))
plt.bar(x - 0.2, scores["Train Accuracy"], width=0.4, label="Train Accuracy")
plt.bar(x + 0.2, scores["Test Accuracy"], width=0.4, label="Test Accuracy")
plt.xticks(x, scores["Model"], rotation=15, ha="right")
plt.ylim(0, 1.05)
plt.title("Train vs Test Accuracy (Underfit vs Baseline vs More Complex)")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()