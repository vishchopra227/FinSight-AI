# 💳 FinSight AI

An AI-powered **Loan Risk Classification Assistant** built using **Python, Scikit-learn, and Streamlit**.

FinSight AI estimates the potential risk of a loan application using borrower income, credit history, debt-to-income ratio, and loan-related features from historical lending data.

> ⚠️ This project is an educational machine learning demonstration and does not make real banking, lending, approval, or rejection decisions.

---

## 🚀 Features

- 📊 Loan risk classification using machine learning
- 💰 Applicant income and loan details analysis
- 📈 Debt-to-income ratio evaluation
- 🏦 Credit history-based risk assessment
- ⚡ Real-time prediction through Streamlit
- 🎯 Estimated risk probability
- 🧹 Automated data preprocessing
- 🔤 Handling of numerical and categorical features
- 💾 Saved trained ML pipeline using Joblib
- 🎨 Interactive dashboard
- 📋 Applicant information summary
- ⚠️ Risk interpretation based on historical patterns

---

## 🖥️ Application Preview

The application allows users to enter applicant and loan details and receive:

- Predicted loan risk category
- Estimated risk probability
- Applicant information summary
- Risk interpretation

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Joblib
- Streamlit
- Jupyter Notebook / VS Code

---

## 📂 Project Structure

```text
FinSight-AI/
│
├── app.py                    # Streamlit web application
├── train_model.py            # Model training and preprocessing
├── requirements.txt          # Python dependencies
├── loans_full_schema.csv     # Historical lending dataset
├── loan_model.pkl            # Saved trained ML pipeline
├── model_metadata.pkl        # Saved feature metadata
├── .gitignore                # Ignored files and folders
└── README.md                 # Project documentation
```

---

## 📊 Dataset

The project uses a historical lending dataset containing borrower, credit, income, and loan-related information.

### Applicant Information

- Annual income
- Employment length
- State
- Homeownership
- Income verification status

### Financial Information

- Debt-to-income ratio
- Annual joint income
- Joint debt-to-income ratio
- Loan amount
- Interest rate
- Monthly installment
- Loan term

### Credit Information

- Delinquencies in the last two years
- Credit inquiries
- Total credit lines
- Open credit lines
- Total credit limit
- Total credit utilized
- Historical failed payments
- Public record bankruptcies
- Credit account information

### Loan Information

- Loan purpose
- Application type
- Loan grade
- Sub-grade
- Issue month
- Initial listing status
- Disbursement method

---

## 🎯 Target Variable

The original dataset contains multiple loan-status categories.

For this project, the target variable is converted into a binary risk classification:

| Original Loan Status | Risk Class |
|---|---|
| Fully Paid | `0 - Lower Risk` |
| Late (16-30 days) | `1 - Potentially Risky` |
| Late (31-120 days) | `1 - Potentially Risky` |
| Charged Off | `1 - Potentially Risky` |

The following statuses are excluded from training because their final outcome is not yet clear:

- Current
- In Grace Period

---

## ⚙️ How It Works

```text
Historical Lending Dataset
          │
          ▼
Data Cleaning
          │
          ▼
Target Variable Creation
          │
          ▼
Feature Selection
          │
          ▼
Missing Value Handling
          │
          ▼
Categorical Feature Encoding
          │
          ▼
Train-Test Split
          │
          ▼
Random Forest Model
          │
          ▼
Model Evaluation
          │
          ▼
Save Trained Pipeline
          │
          ▼
Streamlit Dashboard
          │
          ▼
Loan Risk Prediction
```

---

## 🤖 Machine Learning Model

The current version uses a **Random Forest Classifier**.

Random Forest was selected because it:

- Handles nonlinear relationships
- Works with multiple numerical features
- Captures interactions between financial variables
- Provides probability estimates
- Performs well on structured tabular data

The complete preprocessing and prediction pipeline is saved using Joblib.

```python
joblib.dump(pipeline, "loan_model.pkl")
```

---

## 🧹 Data Preprocessing

The training pipeline performs the following steps:

1. Removes unnecessary index columns.
2. Removes loan statuses with unclear final outcomes.
3. Converts loan status into a binary risk target.
4. Separates numerical and categorical features.
5. Handles missing numerical values using the median.
6. Handles missing categorical values using the most frequent value.
7. Converts categorical features using One-Hot Encoding.
8. Splits the data into training and testing sets.
9. Trains the Random Forest model.
10. Saves the complete pipeline for future predictions.

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For loan-risk classification, **recall and ROC-AUC are especially important**, because identifying potentially risky loans is more important than relying only on accuracy.

The current model achieved the following results on the test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 79.46% |
| ROC-AUC | 0.5702 |

> Note: The dataset contains a relatively small number of risky loan records. Therefore, the model should be treated as a demonstration rather than a production-grade credit-risk system.

---

## ▶️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vishchopra227/FinSight-AI.git
```

### 2. Move into the Project Folder

```bash
cd FinSight-AI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train_model.py
```

This will generate:

```text
loan_model.pkl
model_metadata.pkl
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💡 Example

### Input

```text
Annual Income: $60,000
Loan Amount: $10,000
Interest Rate: 12%
Debt-to-Income Ratio: 15%
Homeownership: RENT
Loan Purpose: debt_consolidation
Application Type: individual
```

### Output

```text
Predicted Risk: Potentially Risky Loan
Estimated Risk Probability: 45.88%
```

The output is an estimated probability generated by the trained machine learning model.

---

## 🔐 Data Leakage Prevention

The following repayment-related columns are excluded from model training:

```text
balance
paid_total
paid_principal
paid_interest
paid_late_fees
```

These columns contain information that may become available only after the loan has been issued. Including them could lead to data leakage and unrealistic model performance.

---

## ⚠️ Limitations

- The dataset is relatively small after filtering loan statuses.
- The number of risky loan records is limited.
- The model is not trained on real-time banking data.
- The prediction should not be used for actual loan approval decisions.
- The model does not replace professional financial or credit-risk assessment.
- Model performance may change when tested on a larger and more balanced dataset.

---
