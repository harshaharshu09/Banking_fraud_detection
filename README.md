# 🏦 Banking Fraud Detection & Risk Analytics

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge&logo=scikit-learn" />
<img src="https://img.shields.io/badge/Data%20Analysis-Pandas-purple?style=for-the-badge&logo=pandas" />
<img src="https://img.shields.io/badge/Visualization-Matplotlib-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Git-GitHub-black?style=for-the-badge&logo=github" />

</p>

---

## 🚀 Project Overview

**Banking Fraud Detection & Risk Analytics** is a Machine Learning based project designed to analyze banking transactions and identify potentially fraudulent activities.

The system processes transaction data, analyzes suspicious patterns, applies Machine Learning classification models, and generates:

- 🔍 Fraud / Genuine prediction
- 📊 Fraud probability
- 🎯 Risk score
- 🚦 Risk level
- 📈 Data visualizations
- 🧠 Model performance metrics
- ⚠️ Business action recommendations

The project demonstrates how transaction-level data can be transformed into meaningful fraud detection and risk analytics insights.

---

## 🎯 Objectives

The major objectives of this project are:

- Detect potentially fraudulent banking transactions
- Analyze transaction behavior and patterns
- Perform data cleaning and preprocessing
- Explore fraud-related patterns using EDA
- Prepare features for Machine Learning
- Compare multiple Machine Learning models
- Evaluate model performance
- Calculate fraud probability
- Generate transaction risk scores
- Classify transactions into **LOW, MEDIUM, and HIGH** risk
- Generate visual analytics for fraud investigation

---

## 🧠 Machine Learning Models

The project implements and compares two classification algorithms:

### 1️⃣ Logistic Regression

Used as a baseline classification model to predict whether a transaction is:

- `0 → Genuine`
- `1 → Fraud`

### 2️⃣ Random Forest

A tree-based ensemble Machine Learning algorithm that combines multiple decision trees to improve classification performance.

Random Forest is also used to generate fraud probabilities for risk analysis.

---

# 🔄 Project Workflow

```text
                    🏦 BANKING TRANSACTION DATA
                              │
                              ▼
                    📋 DATA UNDERSTANDING
                              │
                              ▼
                       🧹 DATA CLEANING
                              │
                              ▼
                    📊 EXPLORATORY DATA ANALYSIS
                              │
                              ▼
                     ⚙️ FEATURE ENGINEERING
                              │
                              ▼
                       🔢 DATA ENCODING
                              │
                              ▼
                      ✂️ TRAIN / TEST SPLIT
                              │
                              ▼
                    🤖 MACHINE LEARNING
                       ┌──────┴──────┐
                       ▼             ▼
               Logistic Regression  Random Forest
                       └──────┬──────┘
                              ▼
                     📈 MODEL EVALUATION
                              │
                              ▼
                    🎯 FRAUD PROBABILITY
                              │
                              ▼
                       📊 RISK SCORE
                              │
                              ▼
                   🚦 RISK LEVEL ANALYSIS
                              │
                              ▼
                    ⚠️ BUSINESS ACTION


## 📊 Data Analysis

The project performs Exploratory Data Analysis to understand transaction behavior and identify patterns related to fraudulent transactions.
Analysis includes:
Fraud vs Genuine distribution
Transaction amount analysis
Transaction type analysis
Location-based fraud analysis
Device-based fraud analysis
Correlation analysis
Risk distribution
Risk score distribution
These visualizations help understand how transaction characteristics relate to fraud.

## 🛠️ Technology Stack

| Category         | Technologies       |
| ---------------- | ------------------ |
| Programming      | Python             |
| Data Processing  | Pandas, NumPy      |
| Machine Learning | Scikit-learn       |
| Visualization    | Matplotlib         |
| Version Control  | Git                |
| Repository       | GitHub             |
| Development      | Visual Studio Code |


## 📈 Model Evaluation

The Machine Learning models are evaluated using standard classification metrics.
| Metric           | Purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| Accuracy         | Overall prediction correctness                                          |
| Precision        | Correct fraud predictions among predicted fraud                         |
| Recall           | Fraud cases correctly detected                                          |
| F1-Score         | Balance between precision and recall                                    |
| Confusion Matrix | Shows correct and incorrect classifications                             |
| ROC-AUC          | Measures the model's ability to separate fraud and genuine transactions |

## 🎯 Fraud Risk Analytics

The system converts the predicted fraud probability into a risk score.
Risk Score
Risk Score = Fraud Probability × 100
Example
Fraud Probability = 0.93

Risk Score = 0.93 × 100
           = 93 / 100
Risk Classification
Risk Score	Risk Level
0 – 29	 🟢 LOW
30 – 69	 🟡 MEDIUM
70 – 100 🔴 HIGH

Risk thresholds are configurable and are used as an analytical example in this project.

## 🔎 Transaction Prediction

The project also demonstrates prediction on a new transaction.
Example:
Transaction
     │
     ▼
Machine Learning Model
     │
     ▼
Fraud Probability
     │
     ▼
Risk Score
     │
     ▼
Risk Level
     │
     ▼
Business Action

Example Output
Result          : FRAUD
Fraud Probability: High
Risk Score       : 100 / 100
Risk Level       : HIGH

Business Action:
Further verification / investigation required

The prediction acts as a risk signal for further analysis. It does not automatically represent a banking decision or transaction block.

## 📁 Project Structure

Banking_fraud_project/
│
├── 📂 outputs/
│   │
│   └── 📂 plots/
│       ├── fraud_distribution.png
│       ├── transaction_amount_distribution.png
│       ├── transaction_amount_by_fraud.png
│       ├── transaction_type_fraud.png
│       ├── location_fraud.png
│       ├── device_fraud.png
│       ├── correlation_heatmap.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── risk_distribution.png
│       └── risk_score_distribution.png
│
├── 📄 banking_fraud.csv
├── 🐍 fraud_detection.py
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📖 README.md

## 📊 Project Outputs

The project automatically generates analytical outputs including:

## 📈 Visualizations

Fraud Distribution
Transaction Amount Distribution
Transaction Amount by Fraud
Transaction Type Fraud Analysis
Location Fraud Analysis
Device Fraud Analysis
Correlation Heatmap
Confusion Matrix
ROC Curve
Risk Distribution
Risk Score Distribution

## 📄 Result Files

model_metrics.csv
new_transaction_result.csv
transaction_risk_scores.csv
These outputs provide both visual and tabular insights into model performance and transaction risk.

## ⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/banking-fraud-detection.git
2️⃣ Navigate to the Project
cd banking-fraud-detection
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Project
python fraud_detection.py

## 📦 Requirements

The project uses the following Python libraries:
pandas
numpy
matplotlib
scikit-learn
Install all dependencies using:
pip install -r requirements.txt

## 🧪 Sample Project Flow

A transaction enters the system.
Transaction Amount
        +
Transaction Type
        +
Location
        +
Device
        │
        ▼
Data Processing
        │
        ▼
Machine Learning Model
        │
        ▼
Fraud Prediction
        │
        ▼
Fraud Probability
        │
        ▼
Risk Score
        │
        ▼
Risk Level
        │
        ▼
Further Verification / Investigation

## 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:
Python-based data analysis
Pandas data processing
Data cleaning
Exploratory Data Analysis
Feature engineering
Categorical data encoding
Train/Test splitting
Classification algorithms
Logistic Regression
Random Forest
Model evaluation
Confusion Matrix
ROC-AUC
Fraud probability estimation
Risk scoring
Risk classification
Data visualization
Git and GitHub project management

## ⚠️ Project Limitations

This project is designed as a Machine Learning and risk analytics demonstration.
Real-world banking fraud detection systems may require:
Much larger transaction datasets
Real-time transaction processing
Advanced anomaly detection
Streaming systems
Customer behavioral profiling
Device intelligence
Network and graph analysis
Continuous model monitoring
Strong security and compliance controls
Therefore, the results of this project should be interpreted as an analytical demonstration rather than a production banking fraud prevention system.

## 🔮 Future Enhancements

Possible future improvements include:
🌐 Real-time fraud detection
🤖 Advanced anomaly detection
🧠 Deep Learning based fraud detection
📊 Interactive dashboard using Streamlit
⚡ Real-time transaction streaming
🔐 Advanced authentication and security analysis
📱 API-based fraud prediction
☁️ Cloud deployment
📈 Model monitoring and retraining
🚨 Automated fraud alerts

## 🏗️ End-to-End Architecture


                    USER TRANSACTION
                           │
                           ▼
                  TRANSACTION DATA
                           │
                           ▼
                  DATA PREPROCESSING
                           │
                           ▼
                  FEATURE ENGINEERING
                           │
                           ▼
                   MACHINE LEARNING
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
        LOGISTIC REGRESSION     RANDOM FOREST
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    MODEL EVALUATION
                           │
                           ▼
                  FRAUD PROBABILITY
                           │
                           ▼
                      RISK SCORE
                           │
                           ▼
                     RISK LEVEL
                           │
                           ▼
                 BUSINESS INSIGHT


## 📸 Project Visualizations

The repository contains generated visualizations inside:
outputs/plots/
These plots provide visual insights into transaction behavior, fraud distribution, model performance, and risk analysis.

## 🔐 Responsible Use

This project is intended for educational, analytical, and demonstration purposes.
Fraud predictions should be treated as risk indicators and should be combined with appropriate verification procedures, business rules, and human review before taking consequential actions.

## 👨‍💻 Project

Banking Fraud Detection & Risk Analytics
Machine Learning | Data Analytics | Fraud Detection | Risk Analysis