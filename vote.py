# ===============================
# Voting Ensemble (LogReg + KNN + RandomForest) + Auto KMeans + Interactive Visuals
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
FINAL_CSV = r"C:/Users/L/Downloads/vfinal_results.csv"
CLUSTER_PROFILE_CSV = r"C:/Users/L/Downloads/vpatient_cluster_profiles.csv"
CENTROIDS_CSV = r"C:/Users/L/Downloads/vcluster_centroids.csv"

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

if 'DiabetesStatus' not in df.columns:
    raise ValueError("Target column 'DiabetesStatus' not found in dataset.")

df = df[selected_features + ['DiabetesStatus']].copy()
df['DiabetesStatus'] = df['DiabetesStatus'].map({"No Diabetes": 0, "Diabetes": 1})

categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']

for col in categorical_cols:
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
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)

# ---------------------------
# 3) Define and tune base models
# ---------------------------
print("\n🔄 Tuning base models...")

# Logistic Regression
lr_params = {'C': [0.1, 1, 10], 'solver': ['lbfgs','liblinear']}
lr_gs = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42),
                     lr_params, cv=5, scoring='roc_auc', n_jobs=-1)
lr_gs.fit(X_train_scaled, y_train)
lr_best = lr_gs.best_estimator_
print("✅ Best LR params:", lr_gs.best_params_)

# KNN
knn_params = {'n_neighbors': [3,5,7,9], 'weights': ['uniform','distance']}
knn_gs = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='roc_auc', n_jobs=-1)
knn_gs.fit(X_train_scaled, y_train)
knn_best = knn_gs.best_estimator_
print("✅ Best KNN params:", knn_gs.best_params_)

# Random Forest
rf_params = {'n_estimators': [100,200], 'max_depth': [None,5,10], 'min_samples_split': [2,5]}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                     rf_params, cv=5, scoring='roc_auc', n_jobs=-1)
rf_gs.fit(X_train_scaled, y_train)
rf_best = rf_gs.best_estimator_
print("✅ Best RF params:", rf_gs.best_params_)

# ---------------------------
# 4) Voting Classifier
# ---------------------------
voting = VotingClassifier(
    estimators=[('lr', lr_best), ('knn', knn_best), ('rf', rf_best)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train_scaled, y_train)
print("\n🤝 Voting ensemble ready!")

# ---------------------------
# 5) Cross-validation fold accuracies
# ---------------------------
print("\n📊 Cross-validation (5-fold) accuracies on training set:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    voting.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
    preds = voting.predict(X_train_scaled[val_idx])
    acc = accuracy_score(y_train.iloc[val_idx], preds)
    fold_accuracies.append(acc)

print("Per-fold accuracies:", np.round(fold_accuracies, 4))
print("Mean:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))

fig_folds = go.Figure([go.Bar(x=[f"Fold {i+1}" for i in range(len(fold_accuracies))],
                             y=fold_accuracies, text=np.round(fold_accuracies,4), textposition='auto')])
fig_folds.update_layout(title="Cross-validation fold accuracies (Voting Ensemble)", yaxis_title="Accuracy")
fig_folds.show()

# ---------------------------
# 6) Learning curve
# ---------------------------
print("\n📈 Computing learning curve...")
train_sizes, train_scores, val_scores = learning_curve(
    voting, X_train_scaled, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig_lc = go.Figure()
fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig_lc.update_layout(title='Learning Curve - Voting Ensemble',
                     xaxis_title='Training examples', yaxis_title='Accuracy')
fig_lc.show()

# ---------------------------
# 7) Final evaluation
# ---------------------------
print("\n🔍 Evaluating on test set...")
proba_test = voting.predict_proba(X_test_scaled)[:,1]
pred_test = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test, pred_test)
prec = precision_score(y_test, pred_test)
rec = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)
auc = roc_auc_score(y_test, proba_test)
kappa = cohen_kappa_score(y_test, pred_test)
mcc = matthews_corrcoef(y_test, pred_test)
cm = confusion_matrix(y_test, pred_test)

print("\n📊 Test set metrics (Voting Ensemble):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"MCC: {mcc:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred_test, digits=4))
print("Confusion matrix:\n", cm)

cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred:0','Pred:1'], y=['True:0','True:1'], colorscale='Blues', text=cm, texttemplate="%{text}"))
cm_fig.update_layout(title='Confusion Matrix - Voting Ensemble')
cm_fig.show()

fpr, tpr, _ = roc_curve(y_test, proba_test)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Voting (AUC={auc:.3f})'))
roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
roc_fig.update_layout(title='ROC Curve - Voting Ensemble', xaxis_title='FPR', yaxis_title='TPR')
roc_fig.show()

# ---------------------------
# 8) Clustering: Auto KMeans
# ---------------------------
print("\n🔍 Running KMeans validation (silhouette + elbow)...")
sil_scores = []
inertias = []
K_range = list(range(2, 10))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)

best_k = K_range[int(np.argmax(sil_scores))]
print("Silhouette scores:", dict(zip(K_range, np.round(sil_scores,4))))
print("Chosen K (by silhouette):", best_k)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=K_range, y=sil_scores, mode='lines+markers', name='Silhouette'), secondary_y=False)
fig.add_trace(go.Scatter(x=K_range, y=inertias, mode='lines+markers', name='Inertia'), secondary_y=True)
fig.update_layout(title="KMeans Validation", xaxis_title='K')
fig.update_yaxes(title_text="Silhouette", secondary_y=False)
fig.update_yaxes(title_text="Inertia", secondary_y=True)
fig.show()

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['ClusterLabel'] = cluster_labels

centroids_scaled = kmeans.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_unscaled, columns=selected_features).round(4)
centroids_df.index.name = 'Cluster'
centroids_df.to_csv(CENTROIDS_CSV)
print(f"💾 Saved centroids to {CENTROIDS_CSV}")

profile = df.groupby('ClusterLabel')[selected_features + ['DiabetesStatus']].mean().round(4)
profile.to_csv(CLUSTER_PROFILE_CSV)
print(f"💾 Saved cluster profiles to {CLUSTER_PROFILE_CSV}")

# ---------------------------
# 9) PCA visualization
# ---------------------------
pca = PCA(n_components=2)
proj = pca.fit_transform(X_scaled)
df['PCA1'] = proj[:,0]
df['PCA2'] = proj[:,1]
df['Vote_Prob'] = voting.predict_proba(X_scaled)[:,1]

fig = px.scatter(df, x='PCA1', y='PCA2', color='ClusterLabel',
                 hover_data=selected_features + ['DiabetesStatus','Vote_Prob'],
                 title=f"PCA projection (k={best_k})")
fig.update_layout(height=700)
fig.show()

# ---------------------------
# 10) Export final results
# ---------------------------
from sklearn.model_selection import train_test_split as _tts
idx = np.arange(len(X_scaled))
_, test_idx_array, _, _ = _tts(idx, y, test_size=0.2, stratify=y, random_state=42)
test_clusters = df.loc[test_idx_array, 'ClusterLabel'].values

final_results = X_test_unscaled.copy()
final_results['Actual'] = y_test.reset_index(drop=True)
final_results['Predicted'] = pred_test
final_results['Pred_Prob'] = proba_test
final_results['ClusterLabel'] = test_clusters
final_results.to_csv(vFINAL_CSV, index=False)
print(f"\n💾 Final results saved to: {vFINAL_CSV}")

# ---------------------------
# 11) Summary
# ---------------------------
print("\n✅ Pipeline complete.")
print(" - Final CSV:", vFINAL_CSV)
print(" - Cluster profiles:", vCLUSTER_PROFILE_CSV)
print(" - Cluster centroids:", vCENTROIDS_CSV)

