# 🏥 Patient Adherence Prediction using Machine Learning

## Project Overview

This project predicts whether a patient is likely to adhere to prescribed medication using supervised machine learning techniques. The objective is to assist healthcare providers in identifying patients who are at risk of non-adherence, enabling timely interventions and improving treatment outcomes.

The project covers the complete end-to-end machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), feature engineering, model selection, hyperparameter tuning, model evaluation, explainability using SHAP, and deployment through a Streamlit web application.

---

## Problem Statement

Medication non-adherence is a major challenge in healthcare, leading to poor clinical outcomes and increased healthcare costs.

The objective of this project is to build a classification model capable of predicting whether a patient will adhere to the prescribed treatment based on demographic, medical, and socioeconomic attributes.

---

## Dataset Features

The model uses the following patient attributes:

* Age
* Gender
* Education Level
* Medication Type
* Dosage (mg)
* Previous Adherence
* Income
* Condition Severity
* Mental Health Status
* Healthcare Access
* Insurance Coverage
* Comorbidities Count
* Social Support Level

Target Variable

* Adherence

  * 1 → Adherent
  * 0 → Non-Adherent

---

## Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)

* Missing value analysis
* Distribution analysis
* Count plots
* Box plots
* Skewness analysis

### 2. Data Preprocessing

* Custom Outlier Capping
* Power Transformation
* Standard Scaling
* Ordinal Encoding
* One-Hot Encoding
* ColumnTransformer Pipeline

### 3. Feature Engineering

* Numerical preprocessing pipeline
* Ordinal feature encoding
* One-hot encoding for nominal variables
* Pipeline-based preprocessing

### 4. Model Selection

The following classification algorithms were evaluated using RandomizedSearchCV:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine (SVM)

---

## Hyperparameter Optimization

RandomizedSearchCV was used with cross-validation to identify the optimal model and hyperparameters.

Evaluation Metric

* Accuracy Score

Cross Validation

* 5-Fold Cross Validation

---

## Model Evaluation

The trained models were evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix
* ROC Curve
* ROC-AUC Score
* Precision-Recall Curve

---

## Model Explainability

Model interpretability was performed using SHAP (SHapley Additive Explanations).

Explainability includes:

* SHAP Summary Plot
* SHAP Waterfall Plot
* Feature Importance Analysis

These techniques provide both global and local explanations for model predictions.

---

## Best Performing Model

**Random Forest Classifier**

The final deployed model was selected based on cross-validation performance and overall predictive capability.

---

## Deployment

The trained pipeline was serialized using Pickle and deployed using Streamlit.

The deployed application allows users to:

* Enter patient information
* Predict adherence status
* View prediction probabilities
* Interact with a user-friendly interface

---

## Technologies Used

Programming Language

* Python

Libraries

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SHAP
* Streamlit

Development Environment

* Google Colab
* Visual Studio Code

---

## Repository Structure

```
Patient_Adherence_Prediction/

│── app.py
│── patient_Adherence_prediction.pkl
│── train_model.ipynb
│── requirements.txt
│── README.md
```

---

## Future Improvements

* Probability calibration
* Model monitoring
* Cross-validation comparison dashboard
* Explainability dashboard
* Cloud deployment
* Automated retraining pipeline

---

## Key Learning Outcomes

* End-to-end supervised machine learning workflow
* Data preprocessing using Scikit-learn Pipelines
* Hyperparameter tuning using RandomizedSearchCV
* Model evaluation using multiple classification metrics
* Model explainability using SHAP
* Machine learning model deployment using Streamlit
