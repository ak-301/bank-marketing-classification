# 🏦 Bank Marketing Classification

> **Predicting whether a bank client will subscribe to a term deposit using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Banks run telephone marketing campaigns to promote term deposits. Calling every customer is expensive and inefficient. This project builds a machine learning system that **ranks customers by their probability of subscribing**, allowing the bank to target the most promising leads.

- **Problem type:** Binary Classification (`y = yes / no`)
- **Approach:** End-to-end ML pipeline with model comparison and interpretability
- **Goal:** Maximize subscriber detection while optimizing cost

---

## 📊 Model Comparison

| Model               | Accuracy | AUC    | Precision | Recall | F1-Score |
|--------------------|----------|--------|-----------|--------|----------|
| Logistic Regression| 0.8175   | **0.7747** | 0.3294    | **0.5981** | 0.4248   |
| Decision Tree      | 0.8362   | 0.6169 | 0.2978    | 0.3341 | 0.3149   |
| Random Forest      | 0.8860   | 0.7562 | 0.4916    | 0.3470 | 0.4068   |
| **XGBoost**        | **0.8950** | 0.7684 | **0.5666** | 0.2888 | 0.3826 |
| SVM                | 0.8589   | 0.7443 | 0.3974    | 0.4881 | **0.4381** |

---

## 🔍 Metric Explanation

- **Accuracy** → Overall correctness of predictions  
- **AUC (ROC-AUC)** → Ability to distinguish between classes (threshold-independent)  
- **Precision** → Of predicted “yes”, how many are correct (important for cost efficiency)  
- **Recall (Sensitivity)** → Of actual “yes”, how many are captured (important for revenue)  
- **F1-Score** → Balance between precision and recall  

---

## 🎯 Key Insight

- **XGBoost** → Best accuracy and precision  
- **Logistic Regression** → Best recall (captures most subscribers)  
- **SVM** → Best overall balance (highest F1-score)  

👉 Model selection depends on **business objective**, not just accuracy.

---

## 📈 Output Charts

| Chart | Description |
|-------|-------------|
| `chart1_class_distribution.png` | Class imbalance visualization |
| `chart2_eda_distributions.png` | Feature distributions |
| `chart3_correlation_heatmap.png` | Feature correlations |
| `chart4_model_comparison.png` | Model performance comparison |
| `chart5_roc_curves.png` | ROC curves |
| `chart6_threshold_optimization.png` | Threshold trade-off |
| `chart7_confusion_matrix.png` | Prediction breakdown |
| `chart8_feature_importance.png` | Feature importance |
| `chart9_shap_summary.png` | Model explainability |
| `chart10_business_simulation.png` | Business impact |

---

## 🧠 Methodology

### Pipeline
```
Raw Data
   ↓
Remove 'duration' (data leakage)
   ↓
One-hot encoding
   ↓
Feature scaling
   ↓
SMOTE (handle imbalance)
   ↓
Train multiple models
   ↓
Model comparison
   ↓
Threshold optimization
   ↓
SHAP analysis
   ↓
Business simulation
```

---

## ⚙️ Why These Choices?

| Decision | Reason |
|----------|--------|
| Remove `duration` | Prevents data leakage |
| SMOTE | Handles class imbalance (~8:1) |
| Multiple models | Enables objective comparison |
| AUC + Recall focus | Accuracy alone is misleading |
| SHAP | Provides interpretability |
| Threshold tuning | Aligns model with business goals |

---

## 🔍 Key Findings

1. Economic indicators are the strongest predictors  
2. Previous campaign success is highly influential  
3. Contact method impacts outcome  
4. Class imbalance significantly affects evaluation  

---

## 💼 Business Impact

- Improves targeting efficiency  
- Reduces unnecessary calls  
- Increases conversion rates  

Example insight:
> A subset of customers can capture a majority of subscribers, significantly reducing outreach cost.

---

## 📁 Repository Structure

```
bank-marketing-classification/
│
├── bank_marketing_project.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── training_data.csv
│   └── testing_data.csv
│
├── notebooks/
│   └── AMS580_Bank_Marketing_Project.ipynb
│
├── charts/
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
    └── AMS580_Bank_Marketing_Report.docx
```

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
shap
```

---

## 📄 Dataset

Based on the UCI Bank Marketing Dataset.

- Customer demographics  
- Campaign details  
- Economic indicators  

**Target:** Whether a client subscribes (`yes` / `no`)

---

## 📚 References

- UCI Bank Marketing Dataset  
- XGBoost Documentation  
- SHAP Documentation  
- imbalanced-learn (SMOTE)

---

## 📝 License

Academic project — AMS 580
