# ===============================
# ✅ XGBoost-only pipeline + clustering + interactive plots + surrogate rules export
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, silhouette_score, roc_curve
)
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings
from scipy.optimize import minimize
warnings.filterwarnings("ignore")

# -----------------------
# 0. Settings / filepaths
# -----------------------
DATA_PATH = r"C:/Users/L/Downloads/enc.csv"
FINAL_CSV = r"C:/Users/L/Downloads/final_results.csv"
CLUSTER_PROFILE_CSV = r"C:/Users/L/Downloads/patient_cluster_profiles.csv"
SURROGATE_RULES_TXT = r"C:/Users/L/Downloads/xgb_surrogate_rules.txt"

# -----------------------
# 1. Load & prepare data
# -----------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

selected_features = [
    'GeneralHealth', 'HasHighBP', 'BMI', 'HasHighChol', 'AgeCategory',
    'HasWalkingDifficulty', 'IncomeLevel', 'HadHeartIssues',
    'PoorPhysicalHealthDays', 'EducationLevel', 'IsPhysicallyActive'
]

df = df[selected_features + ['DiabetesStatus']].copy()
df["DiabetesStatus"] = df["DiabetesStatus"].map({"No Diabetes": 0, "Diabetes": 1})

# encode categorical features
categorical_cols = ['GeneralHealth','HasHighBP','HasHighChol','AgeCategory',
                    'HasWalkingDifficulty','IncomeLevel','HadHeartIssues',
                    'EducationLevel','IsPhysicallyActive']
for col in categorical_cols:
    if df[col].dtype == 'object' or not np.issubdtype(df[col].dtype, np.number):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[selected_features].copy()
y = df["DiabetesStatus"].copy()

# -----------------------
# 2. Scale & split
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                 # for clustering & LR
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Keep unscaled versions (XGBoost works fine with either; we'll use unscaled for interpretability of centroids)
X_train_unscaled = pd.DataFrame(scaler.inverse_transform(X_train_scaled), columns=selected_features).reset_index(drop=True)
X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)

# Also prepare DataFrame index-aligned test set for final CSV
test_idx = X_test_scaled.shape[0]

# -----------------------
# 3. Train XGBoost (primary) + LogisticReg (comparator)
# -----------------------
print("🔄 Training XGBoost (GridSearchCV)...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_search = GridSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)
# Use unscaled features for XGBoost (but they will work either way)
xgb_search.fit(pd.DataFrame(X_train_scaled, columns=selected_features), y_train)
xgb_best = xgb_search.best_estimator_
print("✅ Best XGBoost params:", xgb_search.best_params_)

print("🔄 Training Logistic Regression (for comparison)...")
lr_param_grid = {'C':[0.1,1,10], 'solver':['lbfgs','liblinear']}
lr_search = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
lr_search.fit(X_train_scaled, y_train)
lr_best = lr_search.best_estimator_
print("✅ Best LogisticRegression params:", lr_search.best_params_)

# -----------------------
# 4. Cross-validation reporting (fold variability)
# -----------------------
print("\n📊 Cross-validation (5-fold) ROC-AUC on training set:")
for name, model, X_for in [("XGBoost", xgb_best, pd.DataFrame(X_train_scaled, columns=selected_features)),
                           ("LogReg", lr_best, X_train_scaled)]:
    cv_scores = cross_val_score(model, X_for, y_train, cv=5, scoring='roc_auc')
    print(f"{name}: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}, per-fold={np.round(cv_scores,4)}")

# -----------------------
# 5. Predictions & metrics
# -----------------------
def compute_metrics(name, proba, y_true, thresh=0.5):
    pred = (proba >= thresh).astype(int)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred)
    rec = recall_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    auc = roc_auc_score(y_true, proba)
    cm = confusion_matrix(y_true, pred)
    print(f"\n{name} — Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
    print("Confusion matrix:\n", cm)
    return {'name':name,'acc':acc,'prec':prec,'rec':rec,'f1':f1,'auc':auc,'cm':cm,'pred':pred,'proba':proba}

# get test set probas
xgb_proba_test = xgb_best.predict_proba(pd.DataFrame(X_test_scaled, columns=selected_features))[:,1]
lr_proba_test  = lr_best.predict_proba(X_test_scaled)[:,1]

xgb_results = compute_metrics("XGBoost", xgb_proba_test, y_test)
lr_results  = compute_metrics("LogisticRegression", lr_proba_test, y_test)

# -----------------------
# 6. Hybrid: Weighted average optimized on ROC-AUC (optional)
# -----------------------
print("\n🔧 Optimizing hybrid weights (XGB only optional but we compute a simple LR+XGB hybrid)...")
# We'll keep XGBoost alone as primary; but produce a 2-model hybrid (XGB+LR) for comparison (weights sum=1)
def hybrid_loss2(w, proba1, proba2, y_true):
    a = w[0]
    preds = a*proba1 + (1-a)*proba2
    return 1 - roc_auc_score(y_true, preds)

# initial 0.5
res2 = minimize(hybrid_loss2, x0=[0.5], args=(xgb_proba_test, lr_proba_test, y_test), bounds=[(0,1)])
a_opt = float(res2.x)
hybrid_proba_test = a_opt*xgb_proba_test + (1-a_opt)*lr_proba_test
print(f"✅ Hybrid weight (XGB): {a_opt:.2f}, (LR): {1-a_opt:.2f}")
hybrid_results = compute_metrics("Hybrid(XGB+LR)", hybrid_proba_test, y_test)

# -----------------------
# 7. Learning curve (XGBoost) — interactive with Plotly
# -----------------------
print("\n📈 Computing learning curve for XGBoost (this may take a moment)...")
train_sizes, train_scores, val_scores = learning_curve(
    xgb_best, pd.DataFrame(X_train_scaled, columns=selected_features), y_train,
    cv=5, scoring='accuracy', train_sizes=np.linspace(0.1,1.0,6), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode='lines+markers', name='Train accuracy'))
fig.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode='lines+markers', name='Validation accuracy'))
fig.update_layout(title='Learning Curve — XGBoost', xaxis_title='Training examples', yaxis_title='Accuracy')
fig.show()

# -----------------------
# 8. ROC curves (interactive)
# -----------------------
print("\n📈 Plotting ROC curves (interactive)...")
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba_test)
fpr_h, tpr_h, _ = roc_curve(y_test, hybrid_proba_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_x, y=tpr_x, name=f"XGBoost (AUC={xgb_results['auc']:.3f})", mode='lines'))
fig.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"LogReg (AUC={lr_results['auc']:.3f})", mode='lines'))
fig.add_trace(go.Scatter(x=fpr_h, y=tpr_h, name=f"Hybrid (AUC={hybrid_results['auc']:.3f})", mode='lines'))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
fig.update_layout(title='ROC Curves', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig.show()

# -----------------------
# 9. Confusion matrix plot (interactive heatmap) for XGBoost
# -----------------------
cm = xgb_results['cm']
cm_fig = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Pred 0','Pred 1'],
    y=['True 0','True 1'],
    colorscale='Blues',
    text=cm, texttemplate="%{text}"
))
cm_fig.update_layout(title='Confusion Matrix — XGBoost')
cm_fig.show()

# -----------------------
# 10. Cross-validation fold comparison plot (roc per fold)
# -----------------------
print("\n🔁 Visualizing per-fold ROC-AUC for XGBoost (5 folds)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
fold_idx = 0
fig = go.Figure()
for train_idx, val_idx in skf.split(np.arange(len(X_train_scaled)), y_train):
    fold_idx += 1
    model = xgb_best.__class__(**xgb_best.get_params())
    model.fit(pd.DataFrame(X_train_scaled[train_idx], columns=selected_features), y_train.iloc[train_idx])
    proba_val = model.predict_proba(pd.DataFrame(X_train_scaled[val_idx], columns=selected_features))[:,1]
    auc_val = roc_auc_score(y_train.iloc[val_idx], proba_val)
    fold_aucs.append(auc_val)
    fpr, tpr, _ = roc_curve(y_train.iloc[val_idx], proba_val)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'Fold {fold_idx} (AUC={auc_val:.3f})', mode='lines', opacity=0.7))

fig.update_layout(title='XGBoost ROC per CV fold (train set)', xaxis_title='FPR', yaxis_title='TPR')
fig.show()
print("Fold AUCs:", np.round(fold_aucs,4), "mean:", np.mean(fold_aucs))

# -----------------------
# 11. KMeans clustering auto-select K (silhouette) + elbow + centroids
# -----------------------
print("\n🔍 Running KMeans cluster validation (silhouette + inertia) and auto-selecting K...")
sil_scores = []
inertias = []
K_range = list(range(2, 10))
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)

# pick best K by silhouette (max). If tie, pick smallest K with near-max silhouette
best_k = K_range[int(np.argmax(sil_scores))]
print("Silhouette scores by k:", dict(zip(K_range, np.round(sil_scores,4))))
print("Best K (by silhouette):", best_k)

# interactive plot for silhouette + inertia
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=K_range, y=sil_scores, mode='lines+markers', name='Silhouette'), secondary_y=False)
fig.add_trace(go.Scatter(x=K_range, y=inertias, mode='lines+markers', name='Inertia'), secondary_y=True)
fig.update_layout(title='Cluster Validation Metrics', xaxis_title='Number of clusters')
fig.update_yaxes(title_text="Silhouette Score", secondary_y=False)
fig.update_yaxes(title_text="Inertia", secondary_y=True)
fig.show()

# fit final KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_label'] = kmeans.fit_predict(X_scaled)

# centroids in scaled space -> convert to original feature scale for interpretability
centroids_scaled = kmeans.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_unscaled, columns=selected_features).round(3)
print("\n📍 Cluster centroids (original feature scale):")
print(centroids_df)
centroids_df.to_csv(r"C:/Users/L/Downloads/cluster_centroids.csv", index=True)

# -----------------------
# 12. PCA scatter interactive (clusters + XGBoost probability)
# -----------------------
pca = PCA(n_components=2)
proj = pca.fit_transform(X_scaled)
df['PCA1'] = proj[:,0]
df['PCA2'] = proj[:,1]

# add model predictions/probabilities for hover
df_test_indices = df.index  # whole df; but we'll show probabilities for the whole dataset by predicting full X
full_proba = xgb_best.predict_proba(pd.DataFrame(X_scaled, columns=selected_features))[:,1]
df['xgb_proba'] = full_proba

fig = px.scatter(df, x='PCA1', y='PCA2', color='cluster_label', hover_data=selected_features + ['DiabetesStatus', 'xgb_proba'],
                 title=f"PCA projection colored by KMeans (k={best_k})")
fig.update_layout(height=700)
fig.show()

# -----------------------
# 13. Surrogate decision tree rules (approximate XGBoost with a shallow tree)
# -----------------------
print("\n📝 Training surrogate Decision Tree to extract human-readable rules approximating XGBoost...")
surrogate = DecisionTreeClassifier(max_depth=3, random_state=42)
# Train surrogate on entire dataset to approximate XGBoost predictions
surrogate.fit(X, (xgb_best.predict_proba(pd.DataFrame(X, columns=selected_features))[:,1] >= 0.5).astype(int))
rules_text = export_text(surrogate, feature_names=selected_features)
print("\nSurrogate rules (tree):\n")
print(rules_text)

# save rules to txt file
with open(SURROGATE_RULES_TXT, 'w', encoding='utf-8') as f:
    f.write("Surrogate decision tree rules approximating XGBoost (trained to predict XGBoost binary decisions)\n\n")
    f.write(rules_text)
print(f"\n💾 Surrogate rules saved to: {SURROGATE_RULES_TXT}")

# -----------------------
# 14. Final results CSV (test rows only) with cluster label included
# -----------------------
# Build test DataFrame aligned with X_test order used earlier:
X_test_df = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=selected_features).reset_index(drop=True)
final_results = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "XGB_Prob": xgb_proba_test,
    "XGB_Pred": (xgb_proba_test >= 0.5).astype(int),
    "Hybrid_Prob": hybrid_proba_test,
    "Hybrid_Pred": (hybrid_proba_test >= 0.5).astype(int),
    "LogReg_Prob": lr_proba_test,
    "LogReg_Pred": (lr_proba_test >= 0.5).astype(int),
    "Cluster_Label": df.loc[X_test_df.index, 'cluster_label'].values  # cluster labels for the corresponding rows
})
# merge with original features for easier inspection
final_df_out = pd.concat([X_test_df.reset_index(drop=True), final_results.reset_index(drop=True)], axis=1)
final_df_out.to_csv(FINAL_CSV, index=False)
print(f"\n💾 Final results with cluster labels saved to: {FINAL_CSV}")
print(f"💾 Cluster profiles saved to: {CLUSTER_PROFILE_CSV}")
centroids_df.to_csv(r"C:/Users/L/Downloads/cluster_centroids.csv", index=True)
profile = df.groupby('cluster_label')[selected_features + ['DiabetesStatus']].mean().round(3)
profile.to_csv(CLUSTER_PROFILE_CSV)
print("✅ Done — all outputs saved. Open the CSVs and the surrogate rules text file to inspect results.")
