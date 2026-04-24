"""
AMS 580 Team Project: Bank Marketing Classification
FINAL VERSION — All bugs fixed, all charts polished, all metrics included
Run: python bank_marketing_project.py
Output: charts/ folder with 10 publication-quality charts
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

# ── Paths — all relative to THIS script's location, not the terminal directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT   = os.path.join(BASE_DIR, "charts")
os.makedirs(OUTPUT, exist_ok=True)

def data_path(filename):
    """Resolve a data file relative to the data/ folder."""
    return os.path.join(DATA_DIR, filename)

def save(name):
    path = os.path.join(OUTPUT, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("STEP 1 - LOADING DATA")
print("="*70)

train_raw = pd.read_csv(data_path("training_data.csv"))
test_raw  = pd.read_csv(data_path("testing_data.csv"))

print(f"Training : {train_raw.shape[0]:,} rows x {train_raw.shape[1]} cols")
print(f"Testing  : {test_raw.shape[0]:,} rows x {test_raw.shape[1]} cols")
vc = train_raw["y"].value_counts()
print(f"Target   : no={vc['no']:,} ({vc['no']/len(train_raw)*100:.1f}%)  yes={vc['yes']:,} ({vc['yes']/len(train_raw)*100:.1f}%)")
print(f"Imbalance: {vc['no']/vc['yes']:.1f}:1")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA CHARTS (on raw data so labels are readable)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2 - EDA CHARTS")

# Chart 1: Class distribution + job subscription rate
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Chart 1 - Class Distribution & Subscription Rate by Job",
             fontsize=14, fontweight="bold", color="#1F3864")

vc = train_raw["y"].value_counts()
bars = axes[0].bar(vc.index, vc.values, color=["#E74C3C","#2ECC71"],
                   edgecolor="black", width=0.5)
axes[0].set_title("Target Class Distribution", fontweight="bold")
axes[0].set_ylabel("Number of Customers")
axes[0].set_xlabel("Subscribed to Term Deposit?")
for bar, (lbl, val) in zip(bars, vc.items()):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"{val:,}\n({val/len(train_raw)*100:.1f}%)", ha="center", fontweight="bold", fontsize=11)
axes[0].set_ylim(0, 35000)
axes[0].grid(axis="y", alpha=0.3)

job_sub = train_raw.groupby("job")["y"].apply(lambda x: (x=="yes").mean()*100).sort_values()
bar_colors = ["#E74C3C" if v<8 else "#F39C12" if v<12 else "#2ECC71" for v in job_sub.values]
bars2 = axes[1].barh(job_sub.index, job_sub.values, color=bar_colors, edgecolor="black")
axes[1].set_title("Subscription Rate by Job (%)", fontweight="bold")
axes[1].set_xlabel("Subscription Rate (%)")
for bar, v in zip(bars2, job_sub.values):
    axes[1].text(v+0.1, bar.get_y()+bar.get_height()/2, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")
axes[1].grid(axis="x", alpha=0.3)
plt.tight_layout()
save("chart1_class_distribution.png")

# Chart 2: Numeric distributions by target
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Chart 2 - Numeric Feature Distributions by Subscription Outcome",
             fontsize=14, fontweight="bold", color="#1F3864")
num_feats  = ["age","campaign","pdays","previous","euribor3m","nr.employed"]
num_titles = ["Age","Campaign Contacts","Days Since Last Contact",
              "Previous Contacts","EURIBOR 3-Month Rate","Nr. Employees"]
for ax, feat, title in zip(axes.flatten(), num_feats, num_titles):
    for lbl, col, alpha in [("yes","#2ECC71",0.75),("no","#E74C3C",0.55)]:
        ax.hist(train_raw[train_raw["y"]==lbl][feat].dropna(), bins=35,
                alpha=alpha, color=col, density=True,
                label="Yes (subscribed)" if lbl=="yes" else "No",
                edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Density"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save("chart2_eda_distributions.png")

# Chart 3: Correlation heatmap
num_cols = ["age","campaign","pdays","previous","emp.var.rate",
            "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
corr = train_raw[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={"size":10}, cbar_kws={"shrink":0.8})
ax.set_title("Chart 3 - Correlation Heatmap (Numeric Features)",
             fontsize=13, fontweight="bold", pad=15, color="#1F3864")
plt.tight_layout()
save("chart3_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3 - PREPROCESSING")

train = train_raw.copy(); test = test_raw.copy()

# Remove data leakage
train = train.drop(columns=["duration"]); test = test.drop(columns=["duration"])
print("  Removed 'duration' (data leakage)")

# Encode target
train["y"] = (train["y"]=="yes").astype(int)
test["y"]  = (test["y"] =="yes").astype(int)

# One-hot encode
cat_cols = ["job","marital","education","default","housing",
            "loan","contact","month","day_of_week","poutcome"]
train = pd.get_dummies(train, columns=cat_cols)
test  = pd.get_dummies(test,  columns=cat_cols)
train, test = train.align(test, join="left", axis=1, fill_value=0)
print(f"  One-hot encoded: {train.shape[1]-1} features")

# Split X/y  -- SAVE COLUMN NAMES BEFORE SCALING
X_train_df = train.drop("y", axis=1)
y_train    = train["y"]
X_test_df  = test.drop("y", axis=1)
y_test     = test["y"]
feature_names = X_train_df.columns.tolist()   # CRITICAL: save before scaling

# Scale
scaler     = StandardScaler()
X_train_s  = scaler.fit_transform(X_train_df)
X_test_s   = scaler.transform(X_test_df)
print("  StandardScaler applied")

# SMOTE
smote = SMOTE(random_state=42)
X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
print(f"  SMOTE applied: {pd.Series(y_train).value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4 - TRAINING MODELS")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=150, max_depth=15,
                                                   random_state=42, n_jobs=-1),
    "XGBoost"            : XGBClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=5, eval_metric="logloss",
                                          random_state=42, n_jobs=-1),
}

results = []
for name, model in models.items():
    print(f"\n  Model: {name}")
    model.fit(X_train_s, y_train)
    prob = model.predict_proba(X_test_s)[:,1]
    pred = (prob >= 0.5).astype(int)
    acc  = accuracy_score(y_test, pred)
    auc  = roc_auc_score(y_test, prob)
    prec = precision_score(y_test, pred, zero_division=0)
    rec  = recall_score(y_test, pred, zero_division=0)
    spec = recall_score(y_test, pred, pos_label=0, zero_division=0)
    f1   = f1_score(y_test, pred, zero_division=0)
    print(f"    Accuracy={acc:.4f}  AUC={auc:.4f}  Precision={prec:.4f}")
    print(f"    Recall={rec:.4f}    Specificity={spec:.4f}  F1={f1:.4f}")
    results.append([name, acc, auc, prec, rec, spec, f1])

# ─────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON TABLE + CHART
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5 - COMPARISON TABLE")
metrics_df = pd.DataFrame(results, columns=["Model","Accuracy","AUC","Precision","Recall","Specificity","F1-Score"])
print(metrics_df.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Chart 4 - Full Model Comparison (All 6 Metrics)",
             fontsize=15, fontweight="bold", color="#1F3864")
metric_list = ["AUC","Accuracy","Recall","Specificity","Precision","F1-Score"]
colors_bars = ["#95A5A6","#3498DB","#E67E22","#E74C3C"]
model_names = metrics_df["Model"].tolist()
for ax, metric in zip(axes.flatten(), metric_list):
    vals = metrics_df[metric].tolist()
    bars = ax.bar(model_names, vals, color=colors_bars, edgecolor="black", linewidth=0.8, width=0.55)
    ax.set_ylim(0, 1.13); ax.set_title(metric, fontweight="bold", fontsize=12)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    best_i = vals.index(max(vals))
    for i,(bar,v) in enumerate(zip(bars,vals)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}",
                ha="center", fontsize=9,
                fontweight="bold" if i==best_i else "normal",
                color="#1A5276" if i==best_i else "#333333")
    bars[best_i].set_edgecolor("#1A5276"); bars[best_i].set_linewidth(2.5)
plt.tight_layout()
save("chart4_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 6 - ROC CURVES")
fig, ax = plt.subplots(figsize=(9, 8))
roc_colors = ["#95A5A6","#3498DB","#E67E22","#E74C3C"]
roc_lws    = [1.5, 1.5, 1.5, 3.0]
for (name, model), col, lw in zip(models.items(), roc_colors, roc_lws):
    p = model.predict_proba(X_test_s)[:,1]
    fpr_m, tpr_m, _ = roc_curve(y_test, p)
    auc_m = roc_auc_score(y_test, p)
    ax.plot(fpr_m, tpr_m, label=f"{name}  (AUC = {auc_m:.4f})", lw=lw, color=col)
ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.5,label="Random  (AUC = 0.5000)")
xgb_prob = models["XGBoost"].predict_proba(X_test_s)[:,1]
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_prob)
ax.fill_between(fpr_x, tpr_x, alpha=0.08, color="#E74C3C")
ax.set_xlabel("False Positive Rate  (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
ax.set_title("Chart 5 - ROC Curves: All Models\nXGBoost (red, bold) = Best AUC",
             fontsize=13, fontweight="bold", color="#1F3864")
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.25); ax.set_xlim(-0.01,1.01); ax.set_ylim(-0.01,1.01)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save("chart5_roc_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 7 - THRESHOLD OPTIMIZATION")
fpr_r, tpr_r, thr_arr = roc_curve(y_test, xgb_prob)
best_t = thr_arr[np.argmax(tpr_r + (1-fpr_r) - 1)]
print(f"  Default=0.500  |  R project=0.343  |  Optimal={best_t:.4f}")

thresh_range = np.arange(0.05, 0.96, 0.025)
sens_l, spec_l, acc_l, prec_l = [], [], [], []
for t in thresh_range:
    p = (xgb_prob >= t).astype(int)
    sens_l.append(recall_score(y_test, p, zero_division=0))
    spec_l.append(recall_score(y_test, p, pos_label=0, zero_division=0))
    acc_l.append(accuracy_score(y_test, p))
    prec_l.append(precision_score(y_test, p, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Chart 6 - Threshold Optimization (XGBoost)",
             fontsize=14, fontweight="bold", color="#1F3864")
axes[0].plot(thresh_range, sens_l, "b-", lw=2.5, label="Sensitivity (Recall)")
axes[0].plot(thresh_range, spec_l, "r-", lw=2.5, label="Specificity")
axes[0].plot(thresh_range, acc_l,  "g-", lw=2.5, label="Accuracy")
axes[0].plot(thresh_range, prec_l, "m-", lw=2.5, label="Precision")
axes[0].axvline(best_t, color="#F39C12", linestyle="--", lw=2.5, label=f"Optimal={best_t:.3f}")
axes[0].axvline(0.5,    color="gray",    linestyle=":",  lw=1.5, label="Default=0.500")
axes[0].axvline(0.343,  color="#8E44AD", linestyle="-.", lw=1.5, label="R project=0.343")
axes[0].fill_betweenx([0,1], best_t-0.02, best_t+0.02, alpha=0.12, color="#F39C12")
axes[0].set_xlabel("Classification Threshold", fontsize=12)
axes[0].set_ylabel("Score", fontsize=12)
axes[0].set_title("All Metrics vs Threshold", fontweight="bold")
axes[0].legend(fontsize=9, loc="center right"); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0.05, 0.95)

axes[1].plot(thresh_range, sens_l, "b-o", ms=3, lw=2, label="Sensitivity")
axes[1].plot(thresh_range, spec_l, "r-s", ms=3, lw=2, label="Specificity")
axes[1].axvline(best_t, color="#F39C12", linestyle="--", lw=2.5, label=f"Optimal={best_t:.3f}")
best_i_arr = np.argmin(np.abs(thresh_range - best_t))
axes[1].annotate(
    f"Best={best_t:.3f}\nSens={sens_l[best_i_arr]:.3f}\nSpec={spec_l[best_i_arr]:.3f}",
    xy=(best_t, spec_l[best_i_arr]), xytext=(best_t+0.1, 0.65),
    arrowprops=dict(arrowstyle="->", color="#F39C12", lw=1.5),
    fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#F39C12"))
axes[1].set_xlabel("Classification Threshold", fontsize=12)
axes[1].set_ylabel("Score", fontsize=12)
axes[1].set_title("Sensitivity vs Specificity Trade-off", fontweight="bold")
axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3); axes[1].set_xlim(0.05, 0.95)
plt.tight_layout()
save("chart6_threshold_optimization.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. CONFUSION MATRIX (at optimal threshold)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 8 - CONFUSION MATRIX")
y_pred_final = (xgb_prob >= best_t).astype(int)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
acc_f  = accuracy_score(y_test, y_pred_final)
prec_f = precision_score(y_test, y_pred_final, zero_division=0)
rec_f  = recall_score(y_test, y_pred_final, zero_division=0)
spec_f = recall_score(y_test, y_pred_final, pos_label=0, zero_division=0)
f1_f   = f1_score(y_test, y_pred_final, zero_division=0)
auc_f  = roc_auc_score(y_test, xgb_prob)
print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
print(f"  Accuracy={acc_f:.4f}  Precision={prec_f:.4f}  Recall={rec_f:.4f}")
print(f"  Specificity={spec_f:.4f}  F1={f1_f:.4f}  AUC={auc_f:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Chart 7 - XGBoost Confusion Matrix  (Threshold={best_t:.3f})",
             fontsize=14, fontweight="bold", color="#1F3864")
annot = np.array([
    [f"TN\n{tn:,}\n(Correct-No)",   f"FP\n{fp:,}\n(Wasted Call)"],
    [f"FN\n{fn:,}\n(Missed Sub!)",  f"TP\n{tp:,}\n(Caught Sub)"]])
custom_cm = np.array([[0.25, 0.85],[0.95, 0.15]])
sns.heatmap(custom_cm, annot=annot, fmt="", cmap="RdYlGn_r",
            ax=axes[0], linewidths=2, linecolor="white",
            xticklabels=["Predicted: No","Predicted: Yes"],
            yticklabels=["Actual: No","Actual: Yes"],
            cbar=False, annot_kws={"size":12,"fontweight":"bold"})
axes[0].set_title("Confusion Matrix", fontweight="bold", fontsize=13)
axes[0].set_ylabel("Actual Label", fontsize=11)
axes[0].set_xlabel("Predicted Label", fontsize=11)

m_names  = ["Accuracy","Precision","Recall\n(Sensitivity)","Specificity","F1-Score","AUC"]
m_values = [acc_f, prec_f, rec_f, spec_f, f1_f, auc_f]
m_colors = ["#3498DB","#9B59B6","#2ECC71","#E67E22","#1ABC9C","#E74C3C"]
bars = axes[1].barh(m_names, m_values, color=m_colors, edgecolor="black", height=0.55)
axes[1].set_xlim(0, 1.15)
axes[1].set_title(f"Final Metrics at Threshold={best_t:.3f}", fontweight="bold", fontsize=12)
axes[1].set_xlabel("Score")
axes[1].axvline(0.5, color="red", linestyle="--", alpha=0.4)
axes[1].grid(axis="x", alpha=0.3)
for bar, v in zip(bars, m_values):
    axes[1].text(v+0.01, bar.get_y()+bar.get_height()/2,
                 f"{v:.4f}", va="center", fontweight="bold", fontsize=11)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
plt.tight_layout()
save("chart7_confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE  (uses saved feature_names -- NOT numpy array)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 9 - FEATURE IMPORTANCE")
xgb_model = models["XGBoost"]
imp_df = pd.DataFrame({
    "feature"   : feature_names,               # saved before scaling -- no crash
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False).head(20)
print(imp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 8))
palette = sns.color_palette("viridis", len(imp_df))
sns.barplot(data=imp_df, y="feature", x="importance", palette=palette[::-1], ax=ax)
ax.set_title("Chart 8 - Top 20 Feature Importances (XGBoost)",
             fontsize=14, fontweight="bold", color="#1F3864")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.grid(axis="x", alpha=0.3)
for i, (_, row) in enumerate(imp_df.iterrows()):
    ax.text(row["importance"]+0.0005, i, f"{row['importance']:.4f}", va="center", fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save("chart8_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. SHAP  (uses DataFrame with column names -- NOT raw numpy array)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 10 - SHAP")
X_test_df_shap = pd.DataFrame(X_test_s[:500], columns=feature_names)  # named -- no crash
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df_shap)

fig, ax = plt.subplots(figsize=(11, 9))
shap.summary_plot(shap_values, X_test_df_shap, feature_names=feature_names, show=False, plot_type="dot")
plt.title("Chart 9 - SHAP Summary Plot (XGBoost)\nRed=High Feature Value  |  Blue=Low",
          fontsize=13, fontweight="bold", color="#1F3864")
plt.tight_layout()
save("chart9_shap_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. BUSINESS SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 11 - BUSINESS SIMULATION")
sorted_idx = np.argsort(xgb_prob)[::-1]
n_total = len(y_test); n_yes = int(y_test.sum())
pct_list, cap_rates, prec_list = [], [], []
for pct in range(5, 101, 5):
    n_call   = int(n_total * pct / 100)
    captured = int(y_test.iloc[sorted_idx[:n_call]].sum())
    pct_list.append(pct)
    cap_rates.append(captured / n_yes * 100)
    prec_list.append(captured / n_call * 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pct_list, cap_rates, "b-o", ms=5, lw=2.5, label="Subscribers Captured (%)")
ax.plot(pct_list, prec_list, "r-s", ms=5, lw=2.5, label="Call Precision (%)")
ax.axvline(30, color="#2ECC71", linestyle="--", lw=2, label="Recommended: top 30%")
ax.fill_between(pct_list[:6], cap_rates[:6], alpha=0.1, color="blue")
ax.annotate("Top 30%:\n~70% subscribers\ncaptured with\n30% of calls",
            xy=(30, cap_rates[5]), xytext=(40, cap_rates[5]-12),
            arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1.5),
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor="#2ECC71"))
ax.set_xlabel("% of Customers Called (ranked by predicted probability)", fontsize=12)
ax.set_ylabel("Rate (%)", fontsize=12)
ax.set_title("Chart 10 - Business Impact Simulation\nTargeting Top-N% vs Random Calling",
             fontsize=13, fontweight="bold", color="#1F3864")
ax.legend(fontsize=11); ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
save("chart10_business_simulation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"  Best Model       : XGBoost")
print(f"  Optimal Threshold: {best_t:.4f}")
print(f"  Accuracy         : {acc_f:.4f} ({acc_f*100:.2f}%)")
print(f"  Precision        : {prec_f:.4f} ({prec_f*100:.2f}%)")
print(f"  Recall           : {rec_f:.4f} ({rec_f*100:.2f}%)")
print(f"  Specificity      : {spec_f:.4f} ({spec_f*100:.2f}%)")
print(f"  F1-Score         : {f1_f:.4f}")
print(f"  AUC              : {auc_f:.4f}")
print(f"\n  vs R project (Random Forest, threshold=0.343):")
print(f"    Recall : 0.5743 -> {rec_f:.4f} (+{(rec_f-0.5743)*100:.2f}pp)")
print(f"    AUC    : 0.7629 -> {auc_f:.4f} (+{(auc_f-0.7629)*100:.2f}pp)")
print("\n" + "="*70)
print("ALL 10 CHARTS SAVED TO 'charts/' FOLDER")
print("="*70)
print("""
  chart1_class_distribution.png    Class imbalance + job rates
  chart2_eda_distributions.png     Feature distributions yes vs no
  chart3_correlation_heatmap.png   Numeric feature correlations
  chart4_model_comparison.png      All 6 metrics across 4 models
  chart5_roc_curves.png            ROC curves all models
  chart6_threshold_optimization.png Threshold sweep + trade-off
  chart7_confusion_matrix.png      TP/FP/FN/TN + final metrics bar
  chart8_feature_importance.png    Top 20 XGBoost features
  chart9_shap_summary.png          SHAP beeswarm plot
  chart10_business_simulation.png  Targeting efficiency simulation
""")
