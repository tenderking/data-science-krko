# src/config.py
import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJ_ROOT, "data/raw/krkopt.data")
output_dir = "models/"
output_dim = 18
hidden_layers = [(64, 32), (64, 64, 64, 32), (128, 128, 128, 128, 128, 128, 32)]
learning_rate = [0.01, 0.001, 0.0001]
epochs = 600
lambda_values = 0
# Define parameter distributions for XGBoost
param_dist_xgb = {
    "learning_rate": [0.01, 0.1, 0.2, 0.5, 0.7],
    "n_estimators": [30, 40, 20, 40, 60, 80, 200, 300, 400],
    "max_depth": [2, 5, 8, 10, 15],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "lambda": [0.001, 0.1, 0.5, 0.9, 1, 1.1],
    # Adjust scale_pos_weight for imbalance"scale_pos_weight": [1, 2, 3]
}

# Define parameter distributions for RandomForest
param_dist_rf = {
    "n_estimators": [10, 20, 40, 60, 80, 100, 150, 200, 250, 500],
    "max_depth": [20, 25, 30, 50, 100, 250],
    "min_samples_split": [4, 5, 10, 30, 50, 70],
    "min_samples_leaf": [1, 2, 4, 10, 30, 50, 70],
    "criterion": ["gini", "entropy"],  # Experiment with 'entropy'
    "class_weight": ["balanced", None],  # Adjust class weights
}

# Define parameter distributions for DecisionTree[10,20,40,60,80,90,100,130,150,250,350]
param_dist_dt = {
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [10, 20, 40, 60, 80, 90, 100, 130, 150, 250, 350],
    "min_samples_split": [2, 3, 4, 5, 6, 10, 30, 70],
    "min_samples_leaf": [1, 2, 3, 10, 30, 70],
}

# Define parameter distributions for Bagging 'base_estimator__class_weight': ['balanced', None] # Experiment with 'entropy''base_estimator__criterion': ['gini', 'entropy'],
param_dist_bagging = {
    "n_estimators": [10, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500],
    "max_samples": [0.8, 1.0],
    "max_features": [1.0],
    # Adjust class weights
}
