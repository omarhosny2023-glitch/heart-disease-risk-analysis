# Heart Disease Risk Prediction

Machine learning project (CRISP-DM framework) predicting heart disease presence from clinical attributes, on 918 patient records.

## Overview

This project builds and compares five classification models to predict heart disease, with a deliberate focus on recall over raw accuracy — in a medical screening context, missing a true positive costs more than a false alarm.

## Objectives

- Clean and validate a clinical dataset containing physiologically impossible values
- Select the most predictive features using multiple statistical methods
- Train and compare multiple classifiers, and select a final model optimized for recall

## Dataset

918 patient records, 12 clinical variables (age, resting BP, cholesterol, max heart rate, exercise angina, chest pain type, and others).

## Tools & Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy, Statsmodels

## Methodology

- Data quality assessment: identified physiologically impossible values (RestingBP = 0, Cholesterol = 0)
- Outlier treatment via IQR capping, with before/after distribution comparisons
- Feature scaling (z-score) and skewness assessment
- Feature selection: Chi-Squared/Mutual Information, VIF, and Sequential Feature Selection → final 5-feature set
- Class balance check (55.3% / 44.7%) — no oversampling needed
- Trained and cross-validated 5 classifiers (KNN, Naive Bayes, Decision Tree, Bagging, Random Forest) with GridSearchCV tuning
- Evaluation via accuracy, balanced accuracy, precision, recall, F1, ROC-AUC

## Key Insights

- ChestPainType_ASY and ExerciseAngina were the strongest predictors, accounting for over 90% of feature importance in the Random Forest model
- Naive Bayes, Bagging, and Random Forest all reached 79.63% accuracy and 0.8186 F1-score
- The regularized Decision Tree had the highest recall (84.45%) and an AUC of 0.8219, despite slightly lower accuracy
- All 5 continuous predictors passed VIF checks (1.09–1.30) — no multicollinearity

## Results / Outcome

A regularized Decision Tree was selected as the final model, prioritizing recall for the medical screening context, while retaining full rule-based interpretability — a factor clinicians value when reviewing automated predictions.

## Project Structure

```
heart-disease-risk-analysis/
│
├── README.md
├── data/
├── notebooks/
├── src/
├── outputs/
└── requirements.txt
```

## Skills Demonstrated

Data Cleaning, Exploratory Data Analysis, Feature Engineering, Feature Selection, Machine Learning Model Building & Evaluation, Data Visualization
