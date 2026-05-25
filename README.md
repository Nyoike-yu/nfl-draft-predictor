🏈 Player Draft Prediction Pipeline (GCI_UoJ)
Overview
This repository contains a robust, leak-proof machine learning pipeline designed to predict whether a player will be drafted based on their physical attributes, combine drills, and background information.

The pipeline leverages advanced feature engineering, target encoding, and a rank-blended ensemble of LightGBM, CatBoost, and ExtraTrees classifiers. It is specifically optimized to avoid data leakage by using GroupKFold cross-validation based on the draft Year.

Dependencies
Ensure you have Python 3.8+ installed along with the following libraries; if not install: pip install pandas numpy scikit-learn scipy lightgbm catboost optuna
(Note: Optuna is not strictly required to run this script as the best parameters are already injected, but it was used for hyperparameter tuning).

Project Structure
The script expects the data to be located in a specific local directory (C:\Personal projects\GCI_UoJ\). Ensure the following files are present before running:

train.csv: Training dataset containing the Drafted target and Year column.
test.csv: Testing dataset for final predictions.
sample_submission.csv: Template for the final output format.
competition.py: The main execution script.

Pipeline Architecture
1. Feature Engineering
The pipeline generates highly predictive domain-specific features:
a) Physical Composites: Calculates BMI, Jump-to-Weight ratios, Speed/Power composites, and agility differentials.
b) Missingness Tracking: Flags missing drills and calculates a drill_participation_rate, which is often a strong signal in draft data.
c) Positional Z-Scores: Standardizes drill performances (40-yard dash, vertical jump, etc.) within specific position groups. This ensures a 300lb Lineman is compared to other Linemen, not Wide Receivers.
d) Frequency Encoding: Encodes categorical frequencies for School.
e) Missing Value Imputation: Uses KNNImputer (k=5) to handle any remaining missing values after feature generation.

2. Cross-Validation Strategy
To prevent temporal data leakage, the pipeline utilizes GroupKFold (n_splits=5), using Year as the group parameter. This ensures that models are evaluated on completely unseen draft classes, simulating the real-world challenge of predicting a future draft.

3. Leak-Proof Target Encoding
Target encoding for Position, Position_Group, and School is applied strictly inside the cross-validation loop. A smoothing factor (k=20) is used to blend the categorical mean with the global target mean, preventing overfitting on rare categories.

4. Modeling & Hyperparameters
The ensemble consists of three distinct models, tuned prior to execution (e.g., via Optuna):

i) LightGBM (lgb): Highly tuned gradient boosting tree (2031 estimators, low learning rate, tailored regularization).
ii) CatBoost (cat): Gradient boosting designed for categorical features and robust to overfitting.
iii) ExtraTrees (et): A heavily randomized bagging classifier that adds diversity to the boosting models.

5. Rank Blending Ensemble
Instead of directly averaging probabilities (which can be skewed by miscalibrated models), the pipeline converts all Out-Of-Fold (OOF) predictions into percentiles/ranks using scipy.stats.rankdata.
A LogisticRegression model is then trained on these ranks to find the optimal blending weights. The final predictions are clipped between 0 and 1.

How to Run
Verify your directory paths in the script match your local machine in the terminal, or update the pd.read_csv() paths to your current working directory.

Execute the script: python competition.py 
Monitor the console output. The script will output the progress of each fold, the calculated Rank Blending weights, and the Honest Rank-Blended AUC score.

The final predictions will be automatically saved to submission.csv in the specified directory.
