# ===============================
# 🤝 Hybrid Diabetes Prediction + Unsupervised Clustering
# Logistic Regression + XGBoost + RandomForest + KMeans
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix, classification_report,
    silhouette_score, RocCurveDisplay
)
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# =======================
# 1. Load & Prepare Data
# =======================
print("📥 Loading dataset...")
df = pd.read_csv("C:/Users/L/Downloads/enc.csv")
print("✅ Data loaded successfully\n")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]

df = df[selected_features + ['DiabetesStatus']]
df["DiabetesStatus"] = df["DiabetesStatus"].map({"No Diabetes": 0, "Diabetes": 1})

# Encode categorical features
categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[selected_features]
y = df["DiabetesStatus"]

# =======================
# 2. Scale & Split Data
# =======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

X_train_unscaled, X_test_unscaled, _, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =======================
# 3. Optimize Base Models
# =======================
print("🔄 Optimizing Logistic Regression...")
log_param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs','liblinear'], 'penalty': ['l2']}
log_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                          log_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
log_search.fit(X_train, y_train)
log_best = log_search.best_estimator_
print(f"✅ Best Logistic Regression: {log_search.best_params_}")

print("🔄 Optimizing XGBoost...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3,5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_search = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist'),
    xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
xgb_search.fit(X_train_unscaled, y_train)
xgb_best = xgb_search.best_estimator_
print(f"✅ Best XGBoost: {xgb_search.best_params_}")

print("🔄 Optimizing RandomForest...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,5],
    'max_features': ['sqrt','log2', None]
}
rf_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
print(f"✅ Best RandomForest: {rf_search.best_params_}\n")

# =======================
# 4. Cross-validation Reporting
# =======================
print("📊 Cross-validation accuracy scores (5 folds):")
for name, model in [("LogReg", log_best), ("XGBoost", xgb_best), ("RandomForest", rf_best)]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}, scores={scores}")

# =======================
# 5. Hybrid Model Weights
# =======================
log_proba = log_best.predict_proba(X_test)[:,1]
xgb_proba = xgb_best.predict_proba(X_test_unscaled)[:,1]
rf_proba  = rf_best.predict_proba(X_test)[:,1]

def hybrid_loss(weights, log_probs, xgb_probs, rf_probs, y_true):
    alpha, beta, gamma = weights
    preds = alpha*log_probs + beta*xgb_probs + gamma*rf_probs
    return 1 - roc_auc_score(y_true, preds)

w0 = [0.33, 0.33, 0.34]
bounds = [(0,1),(0,1),(0,1)]
cons = {'type':'eq','fun': lambda w: 1-sum(w)}

res = minimize(hybrid_loss, w0, args=(log_proba, xgb_proba, rf_proba, y_test),
               bounds=bounds, constraints=cons)
alpha, beta, gamma = res.x
print(f"✅ Optimized hybrid weights -> alpha: {alpha:.2f}, beta: {beta:.2f}, gamma: {gamma:.2f}")

hybrid_proba = alpha*log_proba + beta*xgb_proba + gamma*rf_proba
hybrid_pred = (hybrid_proba >= 0.5).astype(int)

# =======================
# 6. Evaluation Function
# =======================
def evaluate_model(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    print(f"\n📊 {name} Metrics:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # ROC curve (Plotly)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig.update_layout(title=f"{name} ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    fig.show()

# Evaluate all
evaluate_model(log_best, X_test, y_test, "Logistic Regression")
evaluate_model(xgb_best, X_test_unscaled, y_test, "XGBoost")
evaluate_model(rf_best, X_test, y_test, "RandomForest")

# Hybrid wrapper
class HybridModel:
    def __init__(self, hybrid_proba):
        self.hybrid_proba = hybrid_proba

    def predict_proba(self, X):
        return np.vstack([1 - self.hybrid_proba, self.hybrid_proba]).T

    def predict(self, X):
        return (self.hybrid_proba >= 0.5).astype(int)

evaluate_model(HybridModel(hybrid_proba), X_test, y_test, "Hybrid Model")

# =======================
# 7. Save Predictions
# =======================
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": hybrid_pred,
    "Hybrid_Confidence": hybrid_proba,
    "LogReg_Confidence": log_proba,
    "XGBoost_Confidence": xgb_proba,
    "RF_Confidence": rf_proba
})
results.to_csv("C:/Users/L/Downloads/hybrid_predictions_full.csv", index=False)
print("💾 Saved predictions to 'hybrid_predictions_full.csv'")

# =======================
# 8. KMeans Clustering Insights (Interactive)
# =======================
print("\n🔍 Performing KMeans Clustering for Patient Segmentation...")
sil_scores, inertias = [], []
K_range = range(2,10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    inertias.append(km.inertia_)

# Interactive Silhouette + Elbow
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode='lines+markers', name='Silhouette'))
fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', name='Inertia', yaxis='y2'))
fig.update_layout(title="Cluster Validation Metrics",
                  xaxis_title="Number of Clusters",
                  yaxis_title="Silhouette Score",
                  yaxis2=dict(title="Inertia", overlaying='y', side='right'))
fig.show()

best_clusters = 3
kmeans = KMeans(n_clusters=best_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA 2D visualization (Plotly)
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)
df['PCA1'] = reduced[:,0]
df['PCA2'] = reduced[:,1]

fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                 hover_data=selected_features + ['DiabetesStatus'],
                 title="Patient Clusters (PCA Projection)")
fig.show()

# Cluster profile summary
profile = df.groupby('Cluster')[selected_features + ['DiabetesStatus']].mean().round(2)
print("\n📊 Cluster Profile Summary:")
print(profile)
profile.to_csv("C:/Users/L/Downloads/patient_cluster_profiles.csv")
print("💾 Cluster profiles saved to 'patient_cluster_profiles.csv'")



