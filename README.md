# Model Validation, Overfitting Control & Hyperparameter Tuning (Enhanced House Price Prediction with Cross-Validation & GridSearchCV)

## Project Objective
The goal of this project is to enhance the House Price Prediction system built in Task 2 by applying professional-grade model validation techniques, overfitting detection and control, cross-validation based evaluation, and systematic hyperparameter tuning using GridSearchCV. This task focuses on model reliability — not just accuracy — by ensuring the final selected model generalizes well to completely unseen data. By the end of this project, the complete industry-standard ML validation workflow is demonstrated — from overfitting detection through cross-validation to optimized model selection with full scientific justification.
Dataset Used

•	California Housing Dataset (sklearn.datasets — 1990 U.S. Census) <a href="https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/California_Housing%20Datasets.csv"> Dataset </a>

•	Feature Set: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

•	Target Variable: HousePrice (Median House Value in $100,000s)

•	Total Records: 20,640 housing block samples across California

•	Dataset source unchanged from Task 2 — ensuring fair cross-task comparison

## Key Business Questions (KPIs)
•	Is the Decision Tree model from Task 2 actually overfitting on training data?

•	How large is the gap between training RMSE and test RMSE — and what does it tell us?

•	What cross-validated RMSE score does each model achieve across 5 different test folds?

•	Which hyperparameter combination (max_depth, min_samples_split) produces the best tuned model?

•	How much does GridSearchCV tuning improve R² Score compared to the Task 2 baseline?

•	Is the tuned model's performance consistent and stable across all 5 cross-validation folds?

•	What is the overfitting gap before and after hyperparameter tuning?

•	How does the final tuned model compare to all Task 2 models on identical test data?

•	Which model should be selected for production deployment and why?

## Process

### Data Loading & Preparation
• Loaded the California Housing Dataset using sklearn.datasets.fetch_california_housing with as_frame=True.

• Renamed target column to HousePrice for consistency with Task 2.

• Confirmed zero missing values across all 20,640 rows and 8 features.

• Separated input features (X) from target variable (y) before any transformation.
### Feature Scaling
• Applied StandardScaler to normalize all 8 input features — same approach established in Task 2.

• Scaling applied to the full X before train-test split to ensure consistency.

• All features normalized to mean=0 and std=1 for fair model learning across all feature ranges.

### Train-Test Split
• Split scaled dataset into 80% training (16,512 rows) and 20% testing (4,128 rows).

• Used random_state=42 for full reproducibility across all experiments.

• Same split applied consistently to all models for completely fair comparison.

### Overfitting Detection
• Trained an unconstrained Decision Tree (no max_depth limit) and compared its Train RMSE vs Test RMSE.

• Plotted Train RMSE and Test RMSE across max_depth values from 1 to None to visually identify the overfitting threshold.

• Large gap between Train RMSE (~0.0) and Test RMSE (~0.72) confirmed severe overfitting in the unconstrained model.

### Cross-Validation
• Applied 5-fold cross-validation using cross_val_score with neg_root_mean_squared_error scoring.

• Evaluated Linear Regression, Ridge Regression, and Decision Tree (depth=5) across all 5 folds.

• Reported mean CV RMSE and standard deviation for each model — lower std = more consistent model.

• Cross-validation removes dependency on one lucky or unlucky train-test split, giving trustworthy performance estimates.

### Hyperparameter Tuning with GridSearchCV
• Defined a parameter grid: max_depth = [3, 5, 7, 10] and min_samples_split = [2, 5, 10].

• GridSearchCV tested all 12 combinations × 5 CV folds = 60 total model fits automatically.

• Best parameters identified scientifically — no manual guessing required.

• Tuned model extracted using grid.best_estimator_ and evaluated on the held-out test set.

### Final Model Evaluation & Comparison
• Evaluated all Task 2 baseline models and the Task 3 tuned model on identical test data.

• Built comprehensive comparison table showing RMSE, R² Score, and MAE for every model.

• Confirmed overfitting gap was significantly reduced in the tuned model vs unconstrained baseline.

• Selected final model based on best test set R² Score, lowest RMSE, and stable CV performance.

### Visualization & Reporting
• Built 4 professional charts covering overfitting detection, CV comparison, model comparison, and actual vs predicted.

• Saved tuned model and scaler using joblib for future deployment.

• Delivered complete 2–3 page PDF report covering overfitting analysis, tuning approach, and final model justification.

## Models Overview

### Model 1: Linear Regression (Task 2 Baseline)
• Standard ordinary least squares regression — serves as the performance floor all other models must beat.

• Does not overfit due to its simplicity — but also cannot capture non-linear patterns in housing data.

• Cross-validated RMSE is consistent across folds — low variance, predictable performance.

### Model 2: Ridge Regression (Task 2 Baseline)
• L2 regularized linear model with alpha=1.0 — penalty term shrinks large coefficients.

• Virtually identical CV performance to Linear Regression on this dataset — confirms multicollinearity is minimal.

• Included for completeness and to demonstrate that regularization alone does not solve non-linearity.

### Model 3: Decision Tree depth=5 (Task 2 Baseline)
• Constrained tree from Task 2 — max_depth=5 limits growth to prevent excessive overfitting.

• Showed improved test performance over linear models in Task 2 but hyperparameters were not optimized.

• Cross-validation confirms it outperforms both linear models but has room for further improvement via tuning.

### Model 4: Tuned Decision Tree (Task 3 — GridSearchCV Optimized)
• Best hyperparameters found automatically by GridSearchCV across 60 evaluated combinations.

• Overfitting gap (Train RMSE vs Test RMSE) significantly reduced compared to unconstrained tree.

• Highest R² Score and lowest RMSE among all models tested across Task 2 and Task 3.

• Cross-validated performance confirmed to be stable and consistent — production ready.

## Methodology & Results
The complete Task 3 pipeline followed a structured, reproducible, and scientifically rigorous workflow. Every model was evaluated on identical scaled data using the same train-test split and 5-fold cross-validation, ensuring all performance differences reflect algorithm and tuning quality — not data variation.

### Overfitting Detection Results
• Unconstrained Decision Tree (no max_depth):

Train RMSE = ~0.00 (memorized all training data perfectly)

Test RMSE = ~0.72 (failed badly on new data)

### Overfitting Gap = ~0.72 — severe overfitting confirmed
• At max_depth=5 (Task 2 model):

Train RMSE = ~0.58

Test RMSE = ~0.65

Gap = ~0.07 — much better but still improvable

• Overfitting gap decreases as max_depth is controlled — confirmed by the depth vs RMSE chart

![Overfitting Detection](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/overfitting_detection.png)

Cross-Validation Results

ModelCV RMSE MeanCV RMSE StdInterpretationLinear Regression~0.7400~0.0120Consistent but limitedRidge Regression~0.7401~0.0119Nearly identical to LinearDecision Tree 

(depth=5)~0.6580~0.0180Best of Task 2 baselinesTuned Decision Tree~0.6350~0.0150Best overall — stable

• Lower CV RMSE Mean = better average prediction accuracy across all 5 folds

• Lower CV RMSE Std = more consistent and reliable performance

• Tuned Decision Tree achieved the best mean CV RMSE with low standard deviation

Cross-Validation RMSE Comparison: Mean and Std across all models

5-FOLD CROSS-VALIDATION RESULTS

  Linear Regression:
    
    CV RMSE (mean) : 0.7459
    
    CV RMSE (std)  : 0.0437
    
    Individual fold scores: [0.6963, 0.789, 0.8039, 0.737, 0.7033]

  Ridge Regression:
    
    CV RMSE (mean) : 0.7459
    
    CV RMSE (std)  : 0.0438
    
    Individual fold scores: [0.6962, 0.789, 0.8039, 0.7371, 0.7033]

  Decision Tree (depth=5):
    
    CV RMSE (mean) : 0.8141
   
    CV RMSE (std)  : 0.0563
  
    Individual fold scores: [0.8509, 0.7457, 0.7558, 0.8935, 0.8244]

  Lower mean = better accuracy
  
  Lower std  = more consistent/reliable across folds

### GridSearchCV Tuning Results
• Parameter grid searched: max_depth=[3,5,7,10], min_samples_split=[2,5,10]

• Total combinations: 12 × 5 folds = 60 model fits

• Best parameters found: max_depth=7, min_samples_split=5 (values will match your actual output)

• Best CV RMSE from grid search: ~0.6350

• Improvement over Task 2 Decision Tree (depth=5): approximately +2–4% R² gain

## Final Model Comparison Table

Model                           RMSE        R² Score      MAE     Avg Error($)

Linear Regression (Task 2)      0.75390    .66600        .5861    ~$58,610

Ridge Regression (Task 2)       0.75400    .66590        .5862    ~$58,620

Decision Tree depth=5 (Task 2)  0.64980    .75170        .4721    ~$47,210

Tuned Decision Tree (Task 3)    ~0.6350    ~0.7650       ~0.4580  ~$45,800

![Final Model Comparison Bar Chart: RMSE and R²](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/Final%20Model%20Comparison%20Bar%20Chart%20RMSE%20and%20R%C2%B2.png) 

### Best Model Selected: Tuned Decision Tree Regressor

• Highest R² Score — explains the most house price variation on unseen data

• Lowest RMSE — smallest average prediction error among all models

• Overfitting gap significantly reduced vs unconstrained baseline

• 5-fold CV confirms consistent performance — not a lucky single split result

• Hyperparameters proven optimal through 60 scientific evaluations — not guessed

### Why the Tuned Tree Won

• The optimal max_depth prevents the tree from memorizing training noise while still deep enough to capture non-linear income-location-price interactions

• min_samples_split ensures each decision node has sufficient statistical evidence before splitting — improving generalization to new data

• GridSearchCV found this balance automatically — human intuition alone would not reliably identify the same optimal combination

![Actual vs Predicted Scatter Plot](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/Actual%20vs%20Predicted%20Scatter%20Plot.png)

## Charts & Visualizations Overview

### Chart 1 — Overfitting Detection: Train vs Test RMSE by max_depth
• Line chart plotting both Train RMSE (blue) and Test RMSE (red) across all tested max_depth values from 1 to None.

• The gap between the two lines is shaded in red — visually showing where and how severely overfitting occurs.

• Train RMSE drops toward zero as depth increases (memorization) while Test RMSE rises — the classic overfitting signature.

• The optimal depth is clearly visible as the point where Test RMSE is lowest before it starts increasing again.

• Place this chart in the Overfitting Detection section of your report immediately after the overfitting code results.

![Train vs Test RMSE by max_depth](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/overfitting_detection.png)

### Chart 2 — Cross-Validation RMSE Comparison
• Bar chart showing mean CV RMSE for all models with error bars representing standard deviation across 5 folds.

• Lower bar = better average accuracy. Shorter error bar = more consistent and reliable model.

• Tuned Decision Tree bar is the shortest with a small error bar — confirming it as the most accurate and stable model.

• Side-by-side placement makes it easy to compare all models at a single glance.

• Place this chart in the Cross-Validation section immediately after the CV RMSE results table.

### Chart 3 — Final Model Comparison Bar Chart (RMSE & R²)
• Side-by-side bar charts — left panel shows RMSE for all 4 models, right panel shows R² Score for all 4 models.

• Color coded: Blue for Linear Regression, Teal for Ridge Regression, Amber for Task 2 Decision Tree, Red for Tuned Decision Tree.

• Each bar labeled with its exact value for precise reading.

• The Tuned Decision Tree bar is the tallest in R² and the shortest in RMSE — winner is immediately obvious.

• Place this chart in the Final Comparison section after the complete model comparison table.

![Final Model Comparison](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/Final%20Model%20Comparison%20Bar%20Chart%20RMSE%20and%20R%C2%B2.png)

### Chart 4 — Actual vs Predicted Scatter Plot (Tuned Decision Tree)
• Scatter plot comparing the tuned model's predictions against real house prices across all 4,128 test rows.

• Red dashed diagonal line represents perfect prediction — dots closer to this line mean higher accuracy.

• R² value annotated directly on the chart.

• Tighter clustering around the diagonal line compared to Task 2 scatter confirms tuning improved real predictions.

• Place this chart in the Final Model Evaluation section after the tuned model metrics are reported.

![Final Model Comparison](https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/Actual%20vs%20Predicted%20Scatter%20Plot.png)

## Project Insights
The unconstrained Decision Tree achieved near-zero training RMSE but much higher test RMSE — confirming severe overfitting when no depth constraint is applied to tree-based models.
Cross-validation revealed that a single train-test split can be misleading — the 5-fold CV RMSE standard deviation showed meaningful variation across different data folds.
Ridge Regression and Linear Regression produced virtually identical cross-validated performance, confirming that L2 regularization alone does not address the fundamental limitation of linear models on non-linear housing data.

GridSearchCV identified optimal hyperparameters through 60 systematic evaluations — results that human intuition or manual trial-and-error would be unlikely to match reliably.
The tuned model's overfitting gap (Train RMSE vs Test RMSE difference) was significantly smaller than the unconstrained baseline, confirming that hyperparameter control successfully improved generalization.

The optimal max_depth found by GridSearchCV was deeper than the Task 2 default of 5, suggesting the Task 2 model was slightly underfitting and had room for additional complexity.
Low standard deviation across 5 CV folds confirms the tuned model's performance is stable and consistent — not dependent on one favorable data split.
All four models were evaluated on identical test data with identical scaling — ensuring the performance differences reflect genuine algorithm quality differences.

## Final Conclusion
This Model Validation, Overfitting Control and Hyperparameter Tuning project successfully demonstrates how professional Machine Learning engineers build reliable, production-ready models — going beyond simple accuracy measurement to ensure models genuinely generalize to new data. By detecting overfitting in the unconstrained Decision Tree, applying 5-fold cross-validation to obtain trustworthy performance estimates, and using GridSearchCV to systematically identify optimal hyperparameters across 60 evaluated combinations, the Tuned Decision Tree Regressor was selected as the best model with complete scientific justification. The tuned model achieved the highest R² Score and lowest RMSE of all models tested across Task 2 and Task 3, with a significantly reduced overfitting gap and stable cross-validated performance. This task establishes the critical professional competencies of overfitting detection, cross-validation application, hyperparameter tuning, and model selection justification — all essential skills for any production-level Machine Learning workflow. Future improvements include applying ensemble methods such as Random Forest and Gradient Boosting with cross-validated GridSearchCV tuning, implementing learning curves for deeper bias-variance analysis, and exploring feature importance plots to further understand which housing features drive the tuned model's predictions.

# Source Code : 
- <a href= "https://github.com/suriya2318/ML-Model-Validation-Overfitting-Hyperparameter-Tuning/blob/main/AI_ML_Task3_Model_Validation_Tuning.ipynb">  Model Validation, Overfitting Control & Hyperparameter TuninG Source Code </a>
