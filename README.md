# Model Validation, Overfitting Control & Hyperparameter Tuning (Enhanced House Price Prediction with Cross-Validation & GridSearchCV)

## Project Objective
The goal of this project is to enhance the House Price Prediction system built in Task 2 by applying professional-grade model validation techniques, overfitting detection and control, cross-validation based evaluation, and systematic hyperparameter tuning using GridSearchCV. This task focuses on model reliability — not just accuracy — by ensuring the final selected model generalizes well to completely unseen data. By the end of this project, the complete industry-standard ML validation workflow is demonstrated — from overfitting detection through cross-validation to optimized model selection with full scientific justification.
Dataset Used

•	California Housing Dataset (sklearn.datasets — 1990 U.S. Census)

•	Feature Set: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

•	Target Variable: HousePrice (Median House Value in $100,000s)

•	Total Records: 20,640 housing block samples across California

•	Dataset source unchanged from Task 2 — ensuring fair cross-task comparison

## Key Business Questions (KPIs)

Is the Decision Tree model from Task 2 actually overfitting on training data?
How large is the gap between training RMSE and test RMSE — and what does it tell us?
What cross-validated RMSE score does each model achieve across 5 different test folds?
Which hyperparameter combination (max_depth, min_samples_split) produces the best tuned model?
How much does GridSearchCV tuning improve R² Score compared to the Task 2 baseline?
Is the tuned model's performance consistent and stable across all 5 cross-validation folds?
What is the overfitting gap before and after hyperparameter tuning?
How does the final tuned model compare to all Task 2 models on identical test data?
Which model should be selected for production deployment and why?

Process

Data Loading & Preparation
• Loaded the California Housing Dataset using sklearn.datasets.fetch_california_housing with as_frame=True.
• Renamed target column to HousePrice for consistency with Task 2.
• Confirmed zero missing values across all 20,640 rows and 8 features.
• Separated input features (X) from target variable (y) before any transformation.
Feature Scaling
• Applied StandardScaler to normalize all 8 input features — same approach established in Task 2.
• Scaling applied to the full X before train-test split to ensure consistency.
• All features normalized to mean=0 and std=1 for fair model learning across all feature ranges.
Train-Test Split
• Split scaled dataset into 80% training (16,512 rows) and 20% testing (4,128 rows).
• Used random_state=42 for full reproducibility across all experiments.
• Same split applied consistently to all models for completely fair comparison.
Overfitting Detection
• Trained an unconstrained Decision Tree (no max_depth limit) and compared its Train RMSE vs Test RMSE.
• Plotted Train RMSE and Test RMSE across max_depth values from 1 to None to visually identify the overfitting threshold.
• Large gap between Train RMSE (~0.0) and Test RMSE (~0.72) confirmed severe overfitting in the unconstrained model.
Cross-Validation
• Applied 5-fold cross-validation using cross_val_score with neg_root_mean_squared_error scoring.
• Evaluated Linear Regression, Ridge Regression, and Decision Tree (depth=5) across all 5 folds.
• Reported mean CV RMSE and standard deviation for each model — lower std = more consistent model.
• Cross-validation removes dependency on one lucky or unlucky train-test split, giving trustworthy performance estimates.
Hyperparameter Tuning with GridSearchCV
• Defined a parameter grid: max_depth = [3, 5, 7, 10] and min_samples_split = [2, 5, 10].
• GridSearchCV tested all 12 combinations × 5 CV folds = 60 total model fits automatically.
• Best parameters identified scientifically — no manual guessing required.
• Tuned model extracted using grid.best_estimator_ and evaluated on the held-out test set.
Final Model Evaluation & Comparison
• Evaluated all Task 2 baseline models and the Task 3 tuned model on identical test data.
• Built comprehensive comparison table showing RMSE, R² Score, and MAE for every model.
• Confirmed overfitting gap was significantly reduced in the tuned model vs unconstrained baseline.
• Selected final model based on best test set R² Score, lowest RMSE, and stable CV performance.
Visualization & Reporting
• Built 4 professional charts covering overfitting detection, CV comparison, model comparison, and actual vs predicted.
• Saved tuned model and scaler using joblib for future deployment.
• Delivered complete 2–3 page PDF report covering overfitting analysis, tuning approach, and final model justification.

Models Overview

Model 1: Linear Regression (Task 2 Baseline)
• Standard ordinary least squares regression — serves as the performance floor all other models must beat.
• Does not overfit due to its simplicity — but also cannot capture non-linear patterns in housing data.
• Cross-validated RMSE is consistent across folds — low variance, predictable performance.
Model 2: Ridge Regression (Task 2 Baseline)
• L2 regularized linear model with alpha=1.0 — penalty term shrinks large coefficients.
• Virtually identical CV performance to Linear Regression on this dataset — confirms multicollinearity is minimal.
• Included for completeness and to demonstrate that regularization alone does not solve non-linearity.
Model 3: Decision Tree depth=5 (Task 2 Baseline)
• Constrained tree from Task 2 — max_depth=5 limits growth to prevent excessive overfitting.
• Showed improved test performance over linear models in Task 2 but hyperparameters were not optimized.
• Cross-validation confirms it outperforms both linear models but has room for further improvement via tuning.
Model 4: Tuned Decision Tree (Task 3 — GridSearchCV Optimized)
• Best hyperparameters found automatically by GridSearchCV across 60 evaluated combinations.
• Overfitting gap (Train RMSE vs Test RMSE) significantly reduced compared to unconstrained tree.
• Highest R² Score and lowest RMSE among all models tested across Task 2 and Task 3.
• Cross-validated performance confirmed to be stable and consistent — production ready.

Methodology & Results
The complete Task 3 pipeline followed a structured, reproducible, and scientifically rigorous workflow. Every model was evaluated on identical scaled data using the same train-test split and 5-fold cross-validation, ensuring all performance differences reflect algorithm and tuning quality — not data variation.

Overfitting Detection Results
• Unconstrained Decision Tree (no max_depth):
Train RMSE = ~0.00 (memorized all training data perfectly)
Test RMSE = ~0.72 (failed badly on new data)
Overfitting Gap = ~0.72 — severe overfitting confirmed
• At max_depth=5 (Task 2 model):
Train RMSE = ~0.58
Test RMSE = ~0.65
Gap = ~0.07 — much better but still improvable
• Overfitting gap decreases as max_depth is controlled — confirmed by the depth vs RMSE chart
[INSERT CHART 1 HERE — Overfitting Detection: Train vs Test RMSE by max_depth]
Cross-Validation Results
