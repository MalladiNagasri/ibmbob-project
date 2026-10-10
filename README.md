# 🏦 Loan Default Prediction & Credit Risk Assessment

A full-stack data science project that helps lenders identify high-risk applicants, estimate capital at risk, and recommend lending actions based on predicted default probability — delivered as a live Streamlit dashboard.

---

## 📋 Overview

This project covers the complete credit-risk analytics workflow:

| Stage | What's Done |
|---|---|
| Data Cleaning | Duplicates removed, missing values imputed, outliers filtered, categoricals encoded |
| KPI Computation | Default rate, capital at risk, avg loan % of income, avg interest rate, total applications |
| Risk Driver Analysis | Default rates by grade, intent, home ownership, income band, loan-to-income band |
| Modelling | Logistic Regression + Random Forest with class-imbalance handling (upsampling) |
| Risk Tiers | Low / Medium / High tiers with recommended lending actions |
| Dashboard | 3-page Streamlit app: Executive Overview, Risk Drivers & Trends, Applicant Risk Checker |

---

## 🛠️ Tech Stack

| Component | Library / Version |
|---|---|
| Dashboard | Streamlit 1.35.0 |
| Data manipulation | pandas 2.2.2, numpy 1.26.4 |
| Visualisation | matplotlib 3.9.0, seaborn 0.13.2 |
| Machine learning | scikit-learn 1.5.0 |
| Model persistence | joblib 1.4.2 |
| Language | Python 3.10+ |

---

## 🚀 How to Run

### 1. Clone / download the project

```bash
git clone <repo-url>
cd loan-credit-risk-analytics
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Make sure the dataset is in the project root

The app expects `credit_risk_dataset.csv` in the same directory as `loan_risk_app.py`.

### 5. Launch the dashboard

```bash
streamlit run loan_risk_app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
loan-credit-risk-analytics/
├── loan_risk_app.py           # Single-file: backend + models + Streamlit UI
├── credit_risk_dataset.csv    # Raw dataset (required)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── Loan_Credit_Risk_Report.docx  # Project report
```

---

## 📊 Dataset

**Credit Risk Dataset** — sourced from Kaggle:
🔗 https://www.kaggle.com/datasets/laotse/credit-risk-dataset

| Column | Description |
|---|---|
| `person_age` | Applicant age |
| `person_income` | Annual income ($) |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Lender-assigned grade (A–G) |
| `loan_amnt` | Loan amount requested ($) |
| `loan_int_rate` | Interest rate (%) |
| `loan_status` | **Target** — 1 = Default, 0 = No Default |
| `loan_percent_income` | Loan amount as fraction of annual income |
| `cb_person_default_on_file` | Historical default flag (Y/N) |
| `cb_person_cred_hist_length` | Credit history length (years) |

---

## 📈 Key Results

- **Random Forest ROC-AUC**: ~0.93+
- **Default Rate in dataset**: ~21%
- **Top default predictors**: loan grade, loan percent income, interest rate, prior default on file
- **Risk tiers**: Low (<30% probability) → Approve; Medium (30–60%) → Approve with surcharge; High (>60%) → Reject/manual review

---

## 📄 License

For educational and analytical use. Dataset licence: see Kaggle link above.
