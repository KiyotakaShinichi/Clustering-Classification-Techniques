Diabetes Prediction Hybrid Model

This project implements a hybrid modeling approach for a balanced diabetes classification dataset. It combines classification and clustering techniques to improve predictive performance and interpretability.

Methods Used

Feature Importance: Identify which features contribute most to the predictions.

K-Means Clustering: Determine optimal cluster of patient groups using the Elbow Method and Silhouette Score.

Classification: Logistic Regression, XGBoost, and K-Nearest Neighbors (KNN), with GridSearchCV for hyperparameter tuning.

Ensemble Learning: Implemented Voting, Boosting, and Stacking approaches to improve model robustness.

Model Evaluation:

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

Confusion Matrix and Learning Curves

Optimal Threshold: Calculated using Youden’s J statistic to balance sensitivity and specificity

Intuition / Goal

The hybrid model uses clustering as an additional feature to provide context about patient profiles. Ensemble learning and careful evaluation allow the model to achieve strong predictive performance while remaining interpretable.


