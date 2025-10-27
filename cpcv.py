# ===============================
# 🤝 Enhanced Hybrid Diabetes Prediction + Clustering + SHAP Explanations
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report,
    silhouette_score
)
import shap
import warnings
warnings.filterwarnings("ignore")

# =======================
# 1. Load & Prepare Data
# =======================
print("📥 Loading dataset...")
df = pd.read_csv("C:/Users/L/Downloads/enc.csv")
print("✅ Data loaded successfully!\n")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
    'HeavyDrinker', 'Sex'
]

df = df[selected_features + ['DiabetesStatus']]
df["DiabetesStatus"] = df["DiabetesStatus"].map({"No Diabetes": 0, "Diabetes": 1})

# Encode categorical features
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df[selected_features]
y = df["DiabetesStatus"]

# =======================
# 2. Split Data (Train/Val/Test)
# =======================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Scale features for Logistic Regression and clustering
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)  # full dataset for clustering

# =======================
# 3. Optimize Base Models
# =======================
print("🔄 Optimizing Logistic Regression...")
log_param_grid = {'C': [0.1,1,10], 'solver': ['lbfgs','liblinear'], 'penalty':['l2']}
log_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                          log_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
log_search.fit(X_train_scaled, y_train)
log_best = log_search.best_estimator_
print(f"✅ Best Logistic Regression Params: {log_search.best_params_}")

print("🔄 Optimizing XGBoost...")
xgb_param_grid = {
    'n_estimators':[100,200],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}
xgb_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'),
    xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
xgb_search.fit(X_train, y_train)  # unscaled fine for XGBoost
xgb_best = xgb_search.best_estimator_
print(f"✅ Best XGBoost Params: {xgb_search.best_params_}\n")

# =======================
# 4. Optimize Hybrid Weights
# =======================
log_proba_val = log_best.predict_proba(X_val_scaled)[:,1]
xgb_proba_val = xgb_best.predict_proba(X_val)[:,1]

alphas = np.linspace(0,1,21)
best_auc = 0
best_alpha = 0
for a in alphas:
    b = 1 - a
    hybrid_val = a*log_proba_val + b*xgb_proba_val
    auc = roc_auc_score(y_val, hybrid_val)
    if auc > best_auc:
        best_auc = auc
        best_alpha = a

alpha, beta = best_alpha, 1-best_alpha
print(f"✅ Optimized hybrid weights -> alpha (LogReg): {alpha:.2f}, beta (XGBoost): {beta:.2f}")

# =======================
# 5. Hybrid Prediction on Test Set
# =======================
log_proba_test = log_best.predict_proba(X_test_scaled)[:,1]
xgb_proba_test = xgb_best.predict_proba(X_test)[:,1]
hybrid_proba = alpha*log_proba_test + beta*xgb_proba_test
hybrid_pred = (hybrid_proba>=0.5).astype(int)

# =======================
# 6. Evaluate Hybrid
# =======================
acc = accuracy_score(y_test, hybrid_pred)
prec = precision_score(y_test, hybrid_pred)
rec = recall_score(y_test, hybrid_pred)
f1 = f1_score(y_test, hybrid_pred)
roc = roc_auc_score(y_test, hybrid_proba)
kappa = cohen_kappa_score(y_test, hybrid_pred)
mcc = matthews_corrcoef(y_test, hybrid_pred)

print("📊 Hybrid Model Metrics:")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"ROC-AUC: {roc:.4f}, Cohen's Kappa: {kappa:.4f}, MCC: {mcc:.4f}")
print("Classification Report:\n", classification_report(y_test, hybrid_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, hybrid_pred))

# =======================
# 7. Confusion Matrix Plot
# =======================
cm = confusion_matrix(y_test, hybrid_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes','Diabetes'],
            yticklabels=['No Diabetes','Diabetes'])
plt.title("Confusion Matrix - Hybrid")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =======================
# 8. Feature Importance + SHAP
# =======================
feature_importance = xgb_best.feature_importances_
features_sorted = sorted(zip(selected_features, feature_importance), key=lambda x: x[1], reverse=True)
plt.figure(figsize=(10,6))
feat_names, feat_scores = zip(*features_sorted)
plt.barh(feat_names, feat_scores, color=sns.color_palette("viridis", len(feat_names)))
plt.xlabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# SHAP explanations using TreeExplainer
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=selected_features)

# =======================
# 9. Individual Model Comparison
# =======================
log_pred = log_best.predict(X_test_scaled)
xgb_pred = xgb_best.predict(X_test)
print("\n📈 Individual Model Accuracy:")
print(f"Logistic Regression: {accuracy_score(y_test, log_pred):.4f}")
print(f"XGBoost: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"Hybrid: {acc:.4f}")

# =======================
# 10. Save Predictions
# =======================
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": hybrid_pred,
    "Hybrid_Confidence": hybrid_proba,
    "LogReg_Confidence": log_proba_test,
    "XGBoost_Confidence": xgb_proba_test
})
results.to_csv("C:/Users/L/Downloads/hybrid_predictions_xgb_shap.csv", index=False)
print("💾 Predictions saved.")

# =======================
# 11. Dynamic KMeans Clustering + Cluster Risk
# =======================
print("\n🔍 Performing Dynamic KMeans Clustering...")
best_sil = -1
best_k = 2
for k in range(2,10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    if sil > best_sil:
        best_sil = sil
        best_k = k

print(f"Optimal number of clusters: {best_k}")
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
df[['PCA1','PCA2']] = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title(f"{best_k} Patient Clusters (PCA Projection)")
plt.tight_layout()
plt.show()

# Cluster-wise diabetes risk
cluster_risk = df.groupby('Cluster')['DiabetesStatus'].mean()
print("\n📊 Cluster-wise Diabetes Risk:")
print(cluster_risk)
plt.figure(figsize=(6,4))
sns.barplot(x=cluster_risk.index, y=cluster_risk.values, palette='magma')
plt.ylabel("Average Diabetes Risk")
plt.xlabel("Cluster")
plt.title("Diabetes Risk by Cluster")
plt.tight_layout()
plt.show()

# Cluster profiling
profile = df.groupby('Cluster')[selected_features].mean().round(2)
print("\n📊 Cluster Profiles:")
print(profile)
profile.to_csv("C:/Users/L/Downloads/patient_cluster_profiles_shap.csv")
print("💾 Cluster profiles saved.")

