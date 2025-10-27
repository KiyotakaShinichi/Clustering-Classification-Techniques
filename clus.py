# ===============================
# Single Logistic Regression with Clustering as Reference
# Approaches:
#  A) cluster-as-feature
#  B) per-cluster logistic regressors
#  C) blend LR probability with cluster prior
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings("ignore")

# --- config (adapt paths if needed) ---
DATA_PATH = r"C:/Users/L/Downloads/enc.csv"
OUT_CSV = r"C:/Users/L/Downloads/final_results_cluster_ref.csv"

# --- 1) load + prepare (same fields you used earlier) ---
df = pd.read_csv(DATA_PATH)
selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]
df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

# label-encode categorical-ish cols (as strings to be safe)
cat_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
            'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
            'EducationLevel','IsPhysicallyActive']
for c in cat_cols:
    df[c] = df[c].astype(str)
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

X = df[selected_features].copy()
y = df['DiabetesStatus'].copy()

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split (same seed as yours)
X_train_scaled, X_test_scaled, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, np.arange(len(X_scaled)), test_size=0.2, stratify=y, random_state=42
)

# --- 2) KMeans clustering (gatekeeper). Keep k=2 as you found best earlier ---
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels_full = kmeans.fit_predict(X_scaled)  # labels for entire dataset
# get train/test cluster labels
cluster_train = cluster_labels_full[idx_train]
cluster_test = cluster_labels_full[idx_test]

# cluster prior (P(y=1 | cluster)) computed on train split only to avoid leakage
cluster_priors = {}
for cl in range(k):
    mask = (cluster_train == cl)
    if mask.sum() == 0:
        cluster_priors[cl] = 0.0
    else:
        cluster_priors[cl] = y_train.reset_index(drop=True)[mask].mean()

print("Cluster priors (P(diabetes|cluster) estimated on train):", cluster_priors)

# -------------------------------
# Approach A: cluster as a feature
# -------------------------------
print("\n--- Approach A: cluster-as-feature (single LR) ---")
# build feature matrix with cluster label appended
X_train_A = np.concatenate([X_train_scaled, cluster_train.reshape(-1,1)], axis=1)
X_test_A  = np.concatenate([X_test_scaled, cluster_test.reshape(-1,1)], axis=1)

lrA = LogisticRegression(max_iter=2000, random_state=42)
# small grid to tune C
gsA = GridSearchCV(lrA, param_grid={'C':[0.01, 0.1,1,10]}, scoring='roc_auc', cv=5, n_jobs=-1)
gsA.fit(X_train_A, y_train)
lrA_best = gsA.best_estimator_
print("Best C (A):", gsA.best_params_)

probaA = lrA_best.predict_proba(X_test_A)[:,1]
predA = (probaA >= 0.5).astype(int)

def print_metrics(y_true, y_pred, y_proba, label):
    print(f"\nMetrics - {label}")
    print("Accuracy:", round(accuracy_score(y_true, y_pred),4))
    print("Precision:", round(precision_score(y_true, y_pred),4))
    print("Recall:", round(recall_score(y_true, y_pred),4))
    print("F1:", round(f1_score(y_true, y_pred),4))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_proba),4))
    print("Cohen's Kappa:", round(cohen_kappa_score(y_true, y_pred),4))
    print("MCC:", round(matthews_corrcoef(y_true, y_pred),4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

print_metrics(y_test, predA, probaA, "LR (cluster feature)")

# -------------------------------
# Approach B: per-cluster LR models
# -------------------------------
print("\n--- Approach B: per-cluster logistic regressors ---")
# train a separate LR for each cluster using train rows only
per_cluster_models = {}
for cl in range(k):
    mask = (cluster_train == cl)
    if mask.sum() < 10:
        per_cluster_models[cl] = None
        continue
    Xc = X_train_scaled[mask]
    yc = y_train.reset_index(drop=True)[mask]
    # tune C lightly
    gs = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42),
                      param_grid={'C':[0.01,0.1,1]}, scoring='roc_auc', cv=3, n_jobs=-1)
    gs.fit(Xc, yc)
    per_cluster_models[cl] = gs.best_estimator_
    print(f" Trained LR for cluster {cl}, best C={gs.best_params_['C']} (train size={mask.sum()})")

# predict with the model of the test instance's cluster
probaB = np.zeros(len(X_test_scaled))
predB = np.zeros(len(X_test_scaled), dtype=int)
for i, cl in enumerate(cluster_test):
    model = per_cluster_models.get(cl)
    if model is None:
        # fallback to global LR trained on all data
        # train global LR if not already
        if 'global_lr' not in locals():
            global_lr = LogisticRegression(max_iter=2000, random_state=42)
            global_lr.fit(X_train_scaled, y_train)
        p = global_lr.predict_proba(X_test_scaled[i].reshape(1,-1))[:,1][0]
    else:
        p = model.predict_proba(X_test_scaled[i].reshape(1,-1))[:,1][0]
    probaB[i] = p
    predB[i] = 1 if p >= 0.5 else 0

print_metrics(y_test, predB, probaB, "Per-cluster LR")

# -------------------------------
# Approach C: blend LR prob with cluster prior
# -------------------------------
print("\n--- Approach C: LR prob blended with cluster prior ---")
# Train a single LR on all training rows (no cluster feature)
lrC = LogisticRegression(max_iter=2000, random_state=42)
gsC = GridSearchCV(lrC, param_grid={'C':[0.01, 0.1, 1, 10]}, scoring='roc_auc', cv=5, n_jobs=-1)
gsC.fit(X_train_scaled, y_train)
lrC_best = gsC.best_estimator_
print("Best C (C):", gsC.best_params_)

proba_lrC = lrC_best.predict_proba(X_test_scaled)[:,1]

# blend parameter alpha: weight for LR prob (0..1)
# final_prob = alpha * lr_prob + (1-alpha) * cluster_prior
def blended_probs(alpha=0.6):
    final_p = np.zeros_like(proba_lrC)
    for i, cl in enumerate(cluster_test):
        prior = cluster_priors.get(int(cl), 0.0)
        final_p[i] = alpha * proba_lrC[i] + (1.0 - alpha) * prior
    return final_p

# try a few alphas and print metrics
for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
    p = blended_probs(alpha=alpha)
    preds = (p >= 0.5).astype(int)
    print(f"\nAlpha={alpha}")
    print_metrics(y_test, preds, p, f"Blended LR (alpha={alpha})")

# choose one alpha to save results (pick alpha=0.7 as example)
alpha_chosen = 0.7
probaC = blended_probs(alpha=alpha_chosen)
predC = (probaC >= 0.5).astype(int)

# -------------------------------
# Save summary/CSV for comparison
# -------------------------------
# Make an output DataFrame with test rows (unscaled features)
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)
out = X_test_unscaled.copy()
out['Actual'] = y_test.reset_index(drop=True)
out['ClusterLabel'] = cluster_test
out['LR_cluster_feature_pred'] = predA
out['LR_cluster_feature_prob'] = probaA
out['PerClusterLR_pred'] = predB
out['PerClusterLR_prob'] = probaB
out['BlendedLR_pred'] = predC
out['BlendedLR_prob'] = probaC

out.to_csv(OUT_CSV, index=False)
print(f"\nSaved results to {OUT_CSV}")

# final print of chosen metrics
print("\nFinal chosen (A/B/C) metrics recap:")
print_metrics(y_test, predA, probaA, "A) LR with cluster feature")
print_metrics(y_test, predB, probaB, "B) Per-cluster LR")
print_metrics(y_test, predC, probaC, f"C) Blended LR (alpha={alpha_chosen})")
