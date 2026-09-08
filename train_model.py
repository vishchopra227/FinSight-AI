import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


DATA_PATH = "loans_full_schema.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# --------------------------------------------------
# 3. Create loan risk target
# --------------------------------------------------

# Low-risk loans
good_statuses = [
    "Fully Paid"
]

# Risky loans: late payment or charged-off loans
risky_statuses = [
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Charged Off"
]

# Keep only finalized or clearly risky loans
df = df[
    df["loan_status"].isin(good_statuses + risky_statuses)
].copy()

# 0 = Low Risk
# 1 = Risky
df["default"] = df["loan_status"].apply(
    lambda status: 1 if status in risky_statuses else 0
)

print("\nLoan risk distribution:")
print(df["default"].value_counts())

print("\nOriginal status distribution:")
print(df["loan_status"].value_counts())

print("\nTarget distribution:")
print(df["default"].value_counts())



features = [
    # Applicant employment and income
    "emp_title",
    "emp_length",
    "state",
    "homeownership",
    "annual_income",
    "verified_income",

    # Debt and income information
    "debt_to_income",
    "annual_income_joint",
    "verification_income_joint",
    "debt_to_income_joint",

    # Credit history
    "delinq_2y",
    "months_since_last_delinq",
    "earliest_credit_line",
    "inquiries_last_12m",
    "total_credit_lines",
    "open_credit_lines",
    "total_credit_limit",
    "total_credit_utilized",
    "num_collections_last_12m",
    "num_historical_failed_to_pay",
    "months_since_90d_late",
    "current_accounts_delinq",
    "total_collection_amount_ever",

    # Account information
    "current_installment_accounts",
    "accounts_opened_24m",
    "months_since_last_credit_inquiry",
    "num_satisfactory_accounts",
    "num_accounts_120d_past_due",
    "num_accounts_30d_past_due",
    "num_active_debit_accounts",
    "total_debit_limit",
    "num_total_cc_accounts",
    "num_open_cc_accounts",
    "num_cc_carrying_balance",
    "num_mort_accounts",
    "account_never_delinq_percent",
    "tax_liens",
    "public_record_bankrupt",

    # Loan details
    "loan_purpose",
    "application_type",
    "loan_amount",
    "term",
    "interest_rate",
    "installment",
    "grade",
    "sub_grade",
    "issue_month",
    "initial_listing_status",
    "disbursement_method"
]



available_features = [
    column for column in features
    if column in df.columns
]

missing_features = [
    column for column in features
    if column not in df.columns
]

if missing_features:
    print("\nMissing features:", missing_features)

X = df[available_features]
y = df["default"]



numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)



model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)




pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_probability = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("Loan Risk Classification Model")
print("----------------")
print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



joblib.dump(
    pipeline,
    "loan_model.pkl"
)

print("\nModel saved successfully as loan_model.pkl")


metadata = {
    "features": available_features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(
    metadata,
    "model_metadata.pkl"
)

print("Metadata saved successfully as model_metadata.pkl")