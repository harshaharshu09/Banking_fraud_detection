# Banking Fraud Detection & Risk Analytics

## Project Overview

This project focuses on detecting potentially fraudulent banking transactions using Machine Learning and analyzing transaction risk.

The system analyzes transaction data, identifies suspicious patterns, predicts whether a transaction is fraudulent or genuine, and generates a fraud probability, risk score, and risk level.

## Objectives

- Detect fraudulent banking transactions
- Analyze transaction patterns
- Compare Machine Learning models
- Evaluate model performance
- Calculate fraud probability
- Generate transaction risk scores
- Classify transactions into LOW, MEDIUM, and HIGH risk levels
- Visualize fraud and transaction patterns

## Machine Learning Models

The project uses:

- Logistic Regression
- Random Forest Classifier

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Git
- GitHub

## Project Workflow

Banking Transaction Data  
↓  
Data Understanding  
↓  
Data Cleaning  
↓  
Exploratory Data Analysis  
↓  
Feature Engineering  
↓  
Data Encoding  
↓  
Train/Test Split  
↓  
Machine Learning Models  
↓  
Model Evaluation  
↓  
Fraud Probability  
↓  
Risk Score  
↓  
Risk Level  
↓  
Business Action

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

## Risk Analysis

The predicted fraud probability is converted into a risk score from 0 to 100.

Risk levels:

- 0–29 → LOW
- 30–69 → MEDIUM
- 70–100 → HIGH

High-risk transactions can be flagged for further verification or investigation.

## Project Outputs

The project generates:

- Fraud distribution plot
- Transaction amount analysis
- Transaction type analysis
- Location-wise fraud analysis
- Device-wise fraud analysis
- Correlation heatmap
- Confusion matrix
- ROC curve
- Risk distribution
- Risk score distribution
- Model metrics CSV
- Transaction risk scores CSV
- New transaction prediction results

## Project Structure

```text
Banking_fraud_project
│
├── outputs/
│   ├── plots/
│   ├── model_metrics.csv
│   ├── new_transaction_result.csv
│   └── transaction_risk_scores.csv
│
├── banking_fraud.csv
├── fraud_detection.py
└── README.md