# 🏦 Bank Marketing Classification

> **Predicting whether a bank client will subscribe to a term deposit using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![AUC](https://img.shields.io/badge/AUC-0.7787-brightgreen)
![Sensitivity](https://img.shields.io/badge/Sensitivity-60.88%25-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Banks run telephone marketing campaigns to promote term deposits. Calling every customer is expensive and inefficient. This project builds a machine learning system that **ranks customers by their probability of subscribing**, allowing the bank to target the most promising leads.

- **Problem type:** Binary Classification (`y = yes / no`)
- **Best model:** XGBoost
- **Key improvement over R baseline:** AUC +1.58pp · Sensitivity +3.45pp

---

## 📊 Results at a Glance

| Metric | R Baseline (Random Forest) | This Project (XGBoost) | Change |
|--------|---------------------------|------------------------|--------|
| **AUC** | 0.7629 | **0.7787** | +1.58pp ✅ |
| **Sensitivity** | 57.43% | **60.88%** | +3.45pp ✅ |
| **Specificity** | 89.00% | 85.52% | Threshold trade-off |
| **Accuracy** | 85.44% | 82.75% | Threshold trade-off |
| **Threshold** | 0.343 | **0.195** | Youden's J optimized |

> **Why lower accuracy is still better:** We intentionally lowered the threshold from 0.343 → 0.195 to capture more subscribers (higher recall). Missing a subscriber = lost revenue. A wasted call = small cost. The trade-off is business-correct.

---

## 📁 Repository Structure

```
bank-marketing-classification/
│
├── bank_marketing_project.py        ← Main script (run this)
├── requirements.txt                 ← All dependencies
├── .gitignore
├── README.md
│
├── data/
│   ├── training_data.csv            ← 32,951 training samples
│   └── testing_data.csv             ← 8,237 test samples
│
├── notebooks/
│   └── AMS580_Bank_Marketing_Project.ipynb  ← Jupyter notebook (step-by-step)
│
├── charts/                          ← All 10 output charts (auto-generated)
│   ├── chart1_class_distribution.png
│   ├── chart2_eda_distributions.png
│   ├── chart3_correlation_heatmap.png
│   ├── chart4_model_comparison.png
│   ├── chart5_roc_curves.png
│   ├── chart6_threshold_optimization.png
│   ├── chart7_confusion_matrix.png
│   ├── chart8_feature_importance.png
│   ├── chart9_shap_summary.png
│   └── chart10_business_simulation.png
│
└── report/
    └── AMS580_Bank_Marketing_Report.docx    ← Full written report
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-classification.git
cd bank-marketing-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main script
```bash
python bank_marketing_project.py
```

All 10 charts will be saved to the `charts/` folder automatically.

### 4. Or open the Jupyter Notebook
```bash
jupyter notebook notebooks/AMS580_Bank_Marketing_Project.ipynb
```

---

## 📈 Output Charts

| Chart | Description |
|-------|-------------|
| `chart1_class_distribution.png` | Class imbalance visualization + subscription rate by job |
| `chart2_eda_distributions.png` | Numeric feature distributions (yes vs no) |
| `chart3_correlation_heatmap.png` | Correlation between numeric features |
| `chart4_model_comparison.png` | All 6 metrics compared across 4 models |
| `chart5_roc_curves.png` | ROC curves with AUC for all models |
| `chart6_threshold_optimization.png` | Threshold sweep — sensitivity/specificity trade-off |
| `chart7_confusion_matrix.png` | TP/FP/FN/TN breakdown + final metrics bar chart |
| `chart8_feature_importance.png` | Top 20 XGBoost feature importances |
| `chart9_shap_summary.png` | SHAP beeswarm — why the model makes each decision |
| `chart10_business_simulation.png` | Subscriber capture vs call volume simulation |

---

## 🧠 Methodology

### Pipeline
```
Raw Data
   ↓
Remove 'duration' (data leakage)
   ↓
One-hot encode 10 categorical features (→ 62 features)
   ↓
StandardScaler (fit on train only)
   ↓
SMOTE (balance training set 1:1)
   ↓
Train 4 models with 10-fold CV
   ↓
Select best model (XGBoost by AUC)
   ↓
Optimize threshold (Youden's J → 0.195)
   ↓
Evaluate + SHAP + Business Simulation
```

### Why these choices?

| Decision | Reason |
|----------|--------|
| Remove `duration` | Only known after the call — data leakage |
| SMOTE | 7.9:1 class imbalance; SMOTE on train only to prevent leakage |
| XGBoost | Best AUC and Sensitivity across 4 models |
| Threshold 0.195 | Youden's J — maximises Sensitivity + Specificity jointly |
| SHAP | Explains individual predictions — not just feature rankings |

---

## 🔍 Key Findings

1. **Economic features dominate** — `nr.employed`, `euribor3m`, `emp.var.rate` are the top 3 predictors. Campaign timing relative to the economic cycle matters more than demographics.

2. **Prior contact success is a strong signal** — `poutcome_success` is the 4th most important feature. A customer who subscribed before will likely do so again.

3. **Cellular > telephone** — Contact channel significantly affects outcome.

4. **Business simulation** — Targeting the top 30% of ranked customers captures ~70% of all subscribers while calling only 30% of the database. That is a **2.3× efficiency gain** over random calling.

---

## 💼 Business Impact Simulation

| Top % Called | Customers Called | Subscribers Captured | Capture Rate | Call Precision |
|-------------|-----------------|---------------------|--------------|----------------|
| 10% | ~824 | ~352 | ~38% | ~42% |
| 20% | ~1,647 | ~519 | ~56% | ~31% |
| **30% ★** | **~2,471** | **~650** | **~70%** | **~26%** |
| 50% | ~4,119 | ~798 | ~86% | ~19% |
| 100% (no model) | 8,237 | 928 | 100% | 11.3% |

★ Recommended operating point — 70% of subscribers captured at 30% of calling cost.

---

## 📦 Dependencies

```
pandas          scikit-learn
numpy           xgboost
matplotlib      imbalanced-learn
seaborn         shap
jupyter
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Dataset

The dataset is based on the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

**Features (20, after removing `duration`):**
- **Client info:** age, job, marital, education, default, housing, loan
- **Campaign:** contact, month, day_of_week, campaign
- **Previous campaign:** pdays, previous, poutcome
- **Economic indicators:** emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

**Target:** `y` — did the client subscribe to a term deposit? (`yes=1`, `no=0`)

---

## 👥 Team

| Member | Contribution |
|--------|-------------|
| Member 1 | EDA, data visualization |
| Member 2 | Preprocessing, SMOTE |
| Member 3 | Model training (LR, DT, RF) |
| Member 4 | XGBoost, threshold optimization |
| Member 5 | SHAP, business simulation, report |

> Replace with actual names before submission.

---

## 📚 References

- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.

---

## 📝 License

This project is for academic purposes — AMS 580, 2026.
