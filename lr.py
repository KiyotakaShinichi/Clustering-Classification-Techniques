# ===============================
# Logistic Regression + Auto KMeans + Interactive Visuals
# - learning curve, per-fold accuracies, confusion matrix, full metrics (incl kappa & MCC)
# - clustering: auto-select K by silhouette, elbow plot, PCA visualization
# - export final_results.csv with cluster label and model confidence
# ===============================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef,
    silhouette_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Config / paths
# ---------------------------
DATA_PATH = r"C:/Users/L/Downloads/enc.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results.csv"
CLUSTER_PROFILE_CSV = r"C:/Users/L/Downloads/patient_cluster_profiles.csv"
CENTROIDS_CSV = r"C:/Users/L/Downloads/cluster_centroids.csv"

# ---------------------------
# 1) Load & prepare data
# ---------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive',
]

# ensure target exists
if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

# Encode categorical features (LabelEncoder as before)
categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
    # handle missing or numeric-like categories safely
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[selected_features].copy()
y = df['DiabetesStatus'].copy()

# ---------------------------
# 2) Scale & split (80/20)
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Keep unscaled test features (original-feature scale) for CSV output & centroids mapping
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)

# ---------------------------
# 3) Logistic Regression (GridSearchCV)
# ---------------------------
print("\n🔄 Tuning Logistic Regression (GridSearchCV)...")
param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs','liblinear'], 'penalty': ['l2']}
lr_gs = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42),
                     param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
lr_gs.fit(X_train_scaled, y_train)
lr = lr_gs.best_estimator_
print("✅ Best params:", lr_gs.best_params_)

# ---------------------------
# 4) Cross-validation fold accuracies (per-fold) - interactive plot
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_idx = 0
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    fold_idx += 1
    lr_clone = LogisticRegression(**lr.get_params())
    lr_clone.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
    preds = lr_clone.predict(X_train_scaled[val_idx])
    acc = accuracy_score(y_train.iloc[val_idx], preds)
    fold_accuracies.append(acc)
print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

# interactive bar chart of fold accuracies
fig_folds = go.Figure([go.Bar(x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
                             y=fold_accuracies, text=np.round(fold_accuracies,4), textposition='auto')])
fig_folds.update_layout(title="Cross-validation fold accuracies (Logistic Regression)",
                        yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 5) Learning curve (interactive)
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    lr, X_train_scaled, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Logistic Regression',
                     xaxis_title='Number of training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 6) Final evaluation on test set & metrics table
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = lr.predict_proba(X_test_scaled)[:,1]
pred_test = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)
kappa = cohen_kappa_score(y_test, pred_test)
mcc = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print("\n📊 Test set metrics (Logistic Regression):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Corr Coef (MCC): {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))
print("Confusion matrix:\n", cm)

# interactive confusion matrix heatmap
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Pred: 0', 'Pred: 1'],
    y=['True: 0', 'True: 1'],
    colorscale='Blues',
    text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix - Logistic Regression (Test set)')
cm_fig.show()

# interactive ROC curve
fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'LR (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Logistic Regression (Test set)', xaxis_title='FPR', yaxis_title='TPR')
roc_fig.show()

# ---------------------------
# 7) Clustering: auto-select K via silhouette + elbow
# ---------------------------
print("\n🔍 Running KMeans validation (silhouette + elbow) to choose K...")
sil_scores = []
inertias = []
K_range = list(range(2, 10))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)

# choose K with max silhouette (tie-break: smallest K)
best_k = K_range[int(np.argmax(sil_scores))]
print("Silhouette scores:", dict(zip(K_range, np.round(sil_scores,4))))
print("Chosen K (by silhouette):", best_k)

# interactive silhouette + elbow plot
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=K_range, y=sil_scores, mode='lines+markers', name='Silhouette'), secondary_y=False)
fig.add_trace(go.Scatter(x=K_range, y=inertias, mode='lines+markers', name='Inertia'), secondary_y=True)
fig.update_layout(title="KMeans validation: Silhouette (left) & Inertia (right)", xaxis_title='K')
fig.update_yaxes(title_text="Silhouette Score", secondary_y=False)
fig.update_yaxes(title_text="Inertia", secondary_y=True)
fig.show()

# fit final KMeans
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['ClusterLabel'] = cluster_labels

# centroids (original feature scale)
centroids_scaled = kmeans.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_unscaled, columns=selected_features).round(4)
centroids_df.index.name = 'Cluster'
print("\n📍 Cluster centroids (original feature scale):")
print(centroids_df)
centroids_df.to_csv(CENTROIDS_CSV)
print(f"💾 Saved centroids to: {CENTROIDS_CSV}")

# cluster profile
profile = df.groupby('ClusterLabel')[selected_features + ['DiabetesStatus']].mean().round(4)
print("\n📊 Cluster profiles (mean values):")
print(profile)
profile.to_csv(CLUSTER_PROFILE_CSV)
print(f"💾 Saved cluster profiles to: {CLUSTER_PROFILE_CSV}")

# ---------------------------
# 8) PCA visualization (interactive)
# ---------------------------
pca = PCA(n_components=2)
proj = pca.fit_transform(X_scaled)
df['PCA1'] = proj[:,0]
df['PCA2'] = proj[:,1]

# probability for full dataset (LR predict_proba)
full_proba = lr.predict_proba(X_scaled)[:,1]
df['LR_Prob'] = full_proba

fig = px.scatter(df, x='PCA1', y='PCA2', color='ClusterLabel',
                 hover_data=selected_features + ['DiabetesStatus', 'LR_Prob'],
                 title=f"PCA projection colored by ClusterLabel (k={best_k})")
fig.update_layout(height=700)
fig.show()

# ---------------------------
# 9) Final CSV: test rows with predictions, prob, cluster
# ---------------------------
# Build DataFrame for test rows: use X_test_unscaled (original-feature scale)
# Map cluster label for each test row: we have df (whole dataset) order corresponds to original df rows;
# we need to retrieve cluster labels for the test rows. The easiest robust way:
# Recompute cluster labels for full scaled X and then pick test rows by their original positions.
# We'll obtain indices of test rows using a split with return indices.

# Re-split to get test indices reproducibly
_, X_test_scaled_for_idx, _, y_test_for_idx = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# cluster_labels array corresponds to rows in df (same order), so find cluster labels for test set rows:
# We must find which rows in the full scaled array belong to the test set used earlier.
# Approach: create a DataFrame of scaled X with original index, then do the same split to get test indices.
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
# perform the same split with indices
from sklearn.model_selection import train_test_split as _tts
idx = np.arange(len(X_scaled_df))
_, test_idx_array, _, _ = _tts(idx, y, test_size=0.2, stratify=y, random_state=42)
test_idx_array = np.array(test_idx_array)

# Now pick rows of X_test_unscaled and associate cluster labels
test_clusters = df.loc[test_idx_array, 'ClusterLabel'].values
# Compose final results
final_results = X_test_unscaled.copy()
final_results = final_results.reset_index(drop=True)
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = pred_test
final_results['Pred_Prob'] = proba_test
final_results['ClusterLabel'] = test_clusters
final_results.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {FINAL_CSV}")

# ---------------------------
# 10) Print summary and finish
# ---------------------------
print("\n✅ Pipeline complete.")
print(" - Final CSV:", FINAL_CSV)
print(" - Cluster profiles:", CLUSTER_PROFILE_CSV)
print(" - Cluster centroids:", CENTROIDS_CSV)
