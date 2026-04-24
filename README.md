# 🏦 Bank Marketing Classification

> Predicting whether a bank client will subscribe to a term deposit using Machine Learning

---

## 📌 Project Overview

Banks run marketing campaigns to promote term deposits. This project builds a machine learning pipeline to **identify high-probability customers**, improving targeting efficiency.

- Binary Classification Problem (`yes / no`)
- Goal: Maximize conversions while reducing unnecessary calls

---

## 📊 Model Comparison

| Model               | Accuracy | AUC    | Precision | Recall | F1-Score |
|--------------------|----------|--------|-----------|--------|----------|
| Logistic Regression| 0.8175   | 0.7747 | 0.3294    | **0.5981** | 0.4248   |
| Decision Tree      | 0.8362   | 0.6169 | 0.2978    | 0.3341 | 0.3149   |
| Random Forest      | 0.8860   | 0.7562 | 0.4916    | 0.3470 | 0.4068   |
| **XGBoost**        | **0.8950** | 0.7684 | **0.5666** | 0.2888 | 0.3826 |
| SVM                | 0.8589   | 0.7443 | 0.3974    | 0.4881 | **0.4381** |

### 🎯 Key Insight
- XGBoost → Best accuracy & precision  
- Logistic → Best recall, SVM → Best balance  

---

## 📈 Chart Insights

### Chart 1 – Class Distribution & Job Impact
- Strong class imbalance (~89% no vs 11% yes) → accuracy alone is misleading  
- Students and retired customers show highest subscription rates  

---

### Chart 2 – Feature Distributions
- Fewer campaign contacts increase subscription probability  
- Economic conditions (interest rates, employment) strongly affect outcomes  

---

### Chart 3 – Correlation Heatmap
- High correlation among economic features (redundant signals)  
- Indicates macroeconomic variables dominate model behavior  

---

### Chart 4 – Model Comparison
- XGBoost leads in accuracy & precision  
- Logistic Regression dominates recall, SVM balances performance  

---

### Chart 5 – ROC Curves
- All models outperform random baseline  
- XGBoost achieves highest AUC (best class separation)  

---

### Chart 6 – Threshold Optimization
- Optimal threshold ≈ 0.206 (not default 0.5)  
- Improves recall significantly while maintaining good specificity  

---

### Chart 7 – Confusion Matrix
- Captures 555 subscribers but misses 373 (recall trade-off)  
- 985 false positives indicate operational cost of outreach  

---

### Chart 8 – Feature Importance
- Top drivers: employment rate, interest rate, campaign features  
- Economic + campaign variables dominate prediction  

---

### Chart 9 – SHAP Analysis
- Confirms importance of economic indicators and contact type  
- Provides interpretability for model decisions  

---

### Chart 10 – Business Simulation
- Top 30% customers capture ~70% of subscribers  
- Targeted strategy significantly outperforms random calling  

---

## 🧠 Methodology

- Removed leakage features (`duration`)  
- One-hot encoding + scaling  
- SMOTE for class imbalance  
- Trained multiple models  
- Threshold tuning + SHAP explainability  

---

## 💼 Business Impact

- Enables **targeted marketing campaigns**  
- Reduces unnecessary calls and cost  
- Improves conversion efficiency  

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
├── charts/
├── notebooks/
└── report/
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

UCI Bank Marketing Dataset  
(Target: subscription yes/no)

---

## 🚀 Final Takeaway

Model performance alone isn’t enough — **threshold tuning + business interpretation** are key to making ML useful in real-world decision-making.
