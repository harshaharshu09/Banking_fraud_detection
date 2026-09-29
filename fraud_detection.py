# ============================================================
# BANKING FRAUD DETECTION AND RISK ANALYTICS
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# -----------------------------
# 2. CREATE OUTPUT FOLDERS
# -----------------------------

os.makedirs("outputs/plots", exist_ok=True)


# -----------------------------
# 3. LOAD DATASET
# -----------------------------

df = pd.read_csv("banking_fraud.csv")

print("\n======================================")
print("BANKING FRAUD DETECTION PROJECT")
print("======================================")

print("\nFirst 5 Transactions:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)


# -----------------------------
# 4. DATA CLEANING
# -----------------------------

print("\n======================================")
print("DATA CLEANING")
print("======================================")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nInvalid Amounts:")
print((df["amount"] <= 0).sum())

print("\nFraud Values:")
print(df["fraud"].unique())

# Remove duplicate rows if any
df = df.drop_duplicates()

# Remove invalid amounts if any
df = df[df["amount"] > 0]

print("\nFinal Dataset Shape:")
print(df.shape)


# -----------------------------
# 5. BASIC FRAUD ANALYSIS
# -----------------------------

print("\n======================================")
print("FRAUD ANALYSIS")
print("======================================")

total_transactions = len(df)

fraud_count = df["fraud"].sum()

genuine_count = total_transactions - fraud_count

fraud_percentage = (fraud_count / total_transactions) * 100

print("Total Transactions:", total_transactions)
print("Genuine Transactions:", genuine_count)
print("Fraud Transactions:", fraud_count)
print("Fraud Percentage:", round(fraud_percentage, 2), "%")


# Average transaction amount

genuine_average = df[df["fraud"] == 0]["amount"].mean()

fraud_average = df[df["fraud"] == 1]["amount"].mean()

print("\nAverage Genuine Transaction:", round(genuine_average, 2))
print("Average Fraud Transaction:", round(fraud_average, 2))


# -----------------------------
# 6. FRAUD BY TRANSACTION TYPE
# -----------------------------

print("\nFraud by Transaction Type:")

fraud_transaction_type = pd.crosstab(
    df["transaction_type"],
    df["fraud"]
)

print(fraud_transaction_type)


# -----------------------------
# 7. FRAUD BY LOCATION
# -----------------------------

print("\nFraud by Location:")

fraud_location = pd.crosstab(
    df["location"],
    df["fraud"]
)

print(fraud_location)


# -----------------------------
# 8. FRAUD BY DEVICE
# -----------------------------

print("\nFraud by Device:")

fraud_device = pd.crosstab(
    df["device"],
    df["fraud"]
)

print(fraud_device)


# ============================================================
# DATA VISUALIZATION
# ============================================================


# -----------------------------
# 9. FRAUD DISTRIBUTION
# -----------------------------

plt.figure(figsize=(7, 5))

df["fraud"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/fraud_distribution.png"
)

plt.close()


# -----------------------------
# 10. TRANSACTION AMOUNT DISTRIBUTION
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["amount"],
    bins=10
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_amount_distribution.png"
)

plt.close()


# -----------------------------
# 11. TRANSACTION AMOUNT BY FRAUD
# -----------------------------

plt.figure(figsize=(7, 5))

df.boxplot(
    column="amount",
    by="fraud"
)

plt.title("Transaction Amount by Fraud Status")
plt.suptitle("")
plt.xlabel("Fraud Status")
plt.ylabel("Transaction Amount")

plt.xticks(
    [1, 2],
    ["Genuine", "Fraud"]
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_amount_by_fraud.png"
)

plt.close()


# -----------------------------
# 12. FRAUD BY TRANSACTION TYPE GRAPH
# -----------------------------

fraud_transaction_type.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Fraud Analysis by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_type_fraud.png"
)

plt.close()


# -----------------------------
# 13. FRAUD BY LOCATION GRAPH
# -----------------------------

fraud_location.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Fraud Analysis by Location")
plt.xlabel("Location")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/plots/location_fraud.png"
)

plt.close()


# -----------------------------
# 14. FRAUD BY DEVICE GRAPH
# -----------------------------

fraud_device.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Fraud Analysis by Device")
plt.xlabel("Device")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/plots/device_fraud.png"
)

plt.close()


# -----------------------------
# 15. CORRELATION HEATMAP
# -----------------------------

correlation_data = df[
    ["amount", "fraud"]
].corr()

plt.figure(figsize=(6, 5))

plt.imshow(
    correlation_data,
    interpolation="nearest"
)

plt.title("Correlation Heatmap")

plt.xticks(
    range(len(correlation_data.columns)),
    correlation_data.columns
)

plt.yticks(
    range(len(correlation_data.columns)),
    correlation_data.columns
)

plt.colorbar()

for i in range(len(correlation_data.columns)):
    for j in range(len(correlation_data.columns)):
        plt.text(
            j,
            i,
            round(correlation_data.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "outputs/plots/correlation_heatmap.png"
)

plt.close()


print("\nAll EDA visualizations created successfully!")


# ============================================================
# MACHINE LEARNING
# ============================================================


# -----------------------------
# 16. CREATE FEATURES
# -----------------------------

print("\n======================================")
print("FEATURE ENGINEERING")
print("======================================")

# Remove transaction_id because it is only an identifier

X = df[
    [
        "amount",
        "transaction_type",
        "location",
        "device"
    ]
]

y = df["fraud"]

print("\nFeature Columns:")
print(X.columns)


# -----------------------------
# 17. ENCODING
# -----------------------------

X_encoded = pd.get_dummies(
    X,
    columns=[
        "transaction_type",
        "location",
        "device"
    ],
    dtype=int
)

print("\nEncoded Features:")
print(X_encoded.head())

print("\nEncoded Feature Columns:")
print(X_encoded.columns)


# -----------------------------
# 18. TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n======================================")
print("TRAIN TEST SPLIT")
print("======================================")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================


# -----------------------------
# 19. LOGISTIC REGRESSION MODEL
# -----------------------------

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train,
    y_train
)

print("\n======================================")
print("LOGISTIC REGRESSION")
print("======================================")

print("Logistic Regression Model Trained Successfully!")


# Predictions

logistic_predictions = logistic_model.predict(
    X_test
)

print("\nLogistic Regression Predictions:")
print(logistic_predictions)

print("\nActual Values:")
print(y_test.values)


# -----------------------------
# 20. LOGISTIC REGRESSION EVALUATION
# -----------------------------

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_probability = logistic_model.predict_proba(
    X_test
)[:, 1]

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probability
)


print("\nLogistic Regression Evaluation:")

print("Accuracy:", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall:", round(logistic_recall, 4))
print("F1 Score:", round(logistic_f1, 4))
print("ROC-AUC:", round(logistic_roc_auc, 4))


# Confusion Matrix

logistic_cm = confusion_matrix(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)


# ============================================================
# RANDOM FOREST
# ============================================================


# -----------------------------
# 21. RANDOM FOREST MODEL
# -----------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

random_forest_model.fit(
    X_train,
    y_train
)

print("\n======================================")
print("RANDOM FOREST")
print("======================================")

print("Random Forest Model Trained Successfully!")


# Predictions

random_forest_predictions = random_forest_model.predict(
    X_test
)

print("\nRandom Forest Predictions:")
print(random_forest_predictions)

print("\nActual Values:")
print(y_test.values)


# -----------------------------
# 22. RANDOM FOREST EVALUATION
# -----------------------------

rf_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

rf_precision = precision_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

rf_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]

rf_roc_auc = roc_auc_score(
    y_test,
    rf_probability
)


print("\nRandom Forest Evaluation:")

print("Accuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_roc_auc, 4))


# Confusion Matrix

rf_cm = confusion_matrix(
    y_test,
    random_forest_predictions
)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)


# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================


plt.figure(figsize=(6, 5))

plt.imshow(
    rf_cm,
    interpolation="nearest"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            rf_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "outputs/plots/confusion_matrix.png"
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================


fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label="Random Forest"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/plots/roc_curve.png"
)

plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================


metrics_data = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],

    "Precision": [
        logistic_precision,
        rf_precision
    ],

    "Recall": [
        logistic_recall,
        rf_recall
    ],

    "F1_Score": [
        logistic_f1,
        rf_f1
    ],

    "ROC_AUC": [
        logistic_roc_auc,
        rf_roc_auc
    ]
})


print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(metrics_data)


metrics_data.to_csv(
    "outputs/model_metrics.csv",
    index=False
)


# ============================================================
# RISK SCORE FOR ALL TRANSACTIONS
# ============================================================


# Random Forest probability for all transactions

all_probabilities = random_forest_model.predict_proba(
    X_encoded
)[:, 1]


# Convert probability to risk score

risk_scores = all_probabilities * 100


# Risk level function

def calculate_risk_level(score):

    if score >= 70:
        return "HIGH"

    elif score >= 30:
        return "MEDIUM"

    else:
        return "LOW"


risk_levels = [
    calculate_risk_level(score)
    for score in risk_scores
]


# Create risk result dataframe

risk_results = df.copy()

risk_results["fraud_probability"] = all_probabilities

risk_results["risk_score"] = risk_scores

risk_results["risk_level"] = risk_levels


print("\n======================================")
print("TRANSACTION RISK ANALYSIS")
print("======================================")

print(
    risk_results[
        [
            "transaction_id",
            "amount",
            "fraud_probability",
            "risk_score",
            "risk_level"
        ]
    ]
)


# Save risk scores

risk_results.to_csv(
    "outputs/transaction_risk_scores.csv",
    index=False
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================


risk_count = risk_results[
    "risk_level"
].value_counts()


plt.figure(figsize=(7, 5))

risk_count.plot(
    kind="bar"
)

plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "outputs/plots/risk_distribution.png"
)

plt.close()


# ============================================================
# RISK SCORE DISTRIBUTION
# ============================================================


plt.figure(figsize=(8, 5))

plt.hist(
    risk_scores,
    bins=10
)

plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/risk_score_distribution.png"
)

plt.close()


# ============================================================
# NEW TRANSACTION PREDICTION
# ============================================================


print("\n======================================")
print("NEW TRANSACTION PREDICTION")
print("======================================")


# New transaction

new_transaction = pd.DataFrame({

    "transaction_id": ["TX209"],

    "amount": [78000],

    "transaction_type": ["Transfer"],

    "location": ["Delhi"],

    "device": ["New"]
})


print("\nNew Transaction:")
print(new_transaction)


# Keep only model features

new_features = new_transaction[
    [
        "amount",
        "transaction_type",
        "location",
        "device"
    ]
]


# Encode new transaction

new_encoded = pd.get_dummies(
    new_features,
    columns=[
        "transaction_type",
        "location",
        "device"
    ],
    dtype=int
)


# Make sure columns match training data

new_encoded = new_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)


# Prediction

new_prediction = random_forest_model.predict(
    new_encoded
)[0]


# Probability

new_probability = random_forest_model.predict_proba(
    new_encoded
)[0][1]


# Risk score

new_risk_score = new_probability * 100


# Risk level

new_risk_level = calculate_risk_level(
    new_risk_score
)


# -----------------------------
# DISPLAY RESULT
# -----------------------------

print("\nNew Transaction Prediction:")

if new_prediction == 1:
    print("Result: FRAUD")
else:
    print("Result: GENUINE")


print("\nNew Transaction Risk Analysis:")

print(
    "Fraud Probability:",
    round(new_probability, 4)
)

print(
    "Risk Score:",
    round(new_risk_score, 2),
    "/ 100"
)

print(
    "Risk Level:",
    new_risk_level
)


# -----------------------------
# BUSINESS ACTION
# -----------------------------

print("\nBusiness Action:")

if new_risk_level == "HIGH":

    print(
        "Action: Further verification / investigation required."
    )

elif new_risk_level == "MEDIUM":

    print(
        "Action: Additional verification recommended."
    )

else:

    print(
        "Action: Transaction appears low risk."
    )


# ============================================================
# SAVE NEW TRANSACTION RESULT
# ============================================================


new_result = pd.DataFrame({

    "transaction_id": ["TX209"],

    "amount": [78000],

    "transaction_type": ["Transfer"],

    "location": ["Delhi"],

    "device": ["New"],

    "fraud_prediction": [
        "FRAUD" if new_prediction == 1 else "GENUINE"
    ],

    "fraud_probability": [
        new_probability
    ],

    "risk_score": [
        new_risk_score
    ],

    "risk_level": [
        new_risk_level
    ]
})


new_result.to_csv(
    "outputs/new_transaction_result.csv",
    index=False
)


# ============================================================
# PROJECT COMPLETED
# ============================================================


print("\n======================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("======================================")

print("\nGenerated Output Files:")

print("1. outputs/model_metrics.csv")

print("2. outputs/transaction_risk_scores.csv")

print("3. outputs/new_transaction_result.csv")

print("4. outputs/plots/fraud_distribution.png")

print("5. outputs/plots/transaction_amount_distribution.png")

print("6. outputs/plots/transaction_amount_by_fraud.png")

print("7. outputs/plots/transaction_type_fraud.png")

print("8. outputs/plots/location_fraud.png")

print("9. outputs/plots/device_fraud.png")

print("10. outputs/plots/correlation_heatmap.png")

print("11. outputs/plots/confusion_matrix.png")

print("12. outputs/plots/roc_curve.png")

print("13. outputs/plots/risk_distribution.png")

print("14. outputs/plots/risk_score_distribution.png")

print("\nAll results saved inside the outputs folder.")