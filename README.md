# 🏥 Patient Adherence Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

![Home Page](images/Home_Page.png)

A supervised machine learning project that predicts whether a patient is likely to **adhere** or **not adhere** to their prescribed medication — helping healthcare providers identify at-risk patients early and intervene before outcomes worsen.

🔗 **Repo:** [AlaSushanth/Patient-Adherence-Prediction](https://github.com/AlaSushanth/Patient-Adherence-Prediction)

---

## 📌 Overview

Medication non-adherence is one of the most costly and preventable problems in healthcare — it drives disease progression, hospital readmissions, and billions of dollars in avoidable costs every year. This project builds an end-to-end ML pipeline that predicts a patient's adherence status from demographic, medical, and socioeconomic attributes, and ships that model as an interactive **Streamlit** web app that a care team can actually use.

The pipeline covers the full lifecycle:

`EDA → Preprocessing → Feature Engineering → Model Selection → Hyperparameter Tuning → Evaluation → Explainability (SHAP) → Deployment`

---

## 🎯 Problem Statement

Build a binary classification model that predicts whether a patient will adhere to their prescribed treatment, using demographic, medical, and socioeconomic attributes — so that healthcare providers can proactively flag and support patients at risk of non-adherence.

**Target Variable — `Adherence`**
| Value | Meaning |
|---|---|
| `1` | Adherent |
| `0` | Non-Adherent |

---

## 🗂️ Dataset Features

| Category | Features |
|---|---|
| Demographic | Age, Gender, Education Level |
| Clinical | Medication Type, Dosage (mg), Condition Severity, Comorbidities Count, Previous Adherence |
| Psychosocial | Mental Health Status, Social Support Level |
| Socioeconomic | Income, Healthcare Access, Insurance Coverage |

---

## 🔬 Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
- Missing value analysis
- Distribution analysis
- Count plots & box plots
- Skewness analysis

### 2. Data Preprocessing
- Custom Outlier Capping
- Power Transformation (for skewed numerical features)
- Standard Scaling
- Ordinal Encoding (for ordered categorical features)
- One-Hot Encoding (for nominal categorical features)
- Unified via a `ColumnTransformer` pipeline

### 3. Feature Engineering
- Numerical preprocessing pipeline
- Ordinal feature encoding pipeline
- One-hot encoding pipeline for nominal variables
- Combined into a single, reusable `Pipeline` object (preprocessing + model)

### 4. Model Selection
The following algorithms were trained and compared using **RandomizedSearchCV**:

| Model |
|---|
| Logistic Regression |
| Decision Tree Classifier |
| Random Forest Classifier ✅ *(best)* |
| Gradient Boosting Classifier |
| Support Vector Machine (SVM) |

---

## ⚙️ Hyperparameter Optimization

- **Search strategy:** `RandomizedSearchCV`
- **Cross-validation:** 5-Fold CV
- **Scoring metric used:** Accuracy

> ⚠️ **Known limitation:** Accuracy was used as the search metric, but with an imbalanced adherence target this can be misleading — ROC-AUC / PR-AUC or F-beta would be a more robust choice for future iterations.

---

## 📊 Model Evaluation

The final model was evaluated using:

- Accuracy Score & Classification Report
- Confusion Matrix
- ROC Curve & ROC-AUC Score
- **Precision-Recall Curve** (below, for the Non-Adherent class)

![Precision-Recall Curve](images/precision_recall_curve.png)

*Average Precision (AP) = 0.73 for the non-adherent class — precision starts high (~0.9) at low recall and gradually declines to ~0.55 as recall approaches 1.0, reflecting the classic recall-vs-false-alarm tradeoff in this problem.*

---

## 🧠 Model Explainability (SHAP)

Model interpretability was performed using **SHAP (SHapley Additive Explanations)**:

- **SHAP Summary Plot** — global feature importance across all predictions
- **SHAP Waterfall Plot** — local, patient-level explanation for a single prediction
- Feature importance ranking, with **Previous Adherence** as the top predictor

These plots make the model's decisions interpretable for clinical stakeholders, not just data scientists.

---

## 🏆 Best Performing Model

**Random Forest Classifier**, selected based on cross-validated performance across accuracy, ROC-AUC, and precision-recall trade-offs relative to the other four candidate models.

---

## 🚀 Deployment

The full preprocessing + model pipeline was serialized with **Pickle** and deployed as an interactive **Streamlit** web application.

The app allows a user to:
- Enter patient information through a simple form
- Get an instant adherence prediction
- View the underlying prediction probability
- Interact through a clean, healthcare-friendly UI

### Run it locally
```bash
git clone https://github.com/AlaSushanth/Patient-Adherence-Prediction.git
cd Patient-Adherence-Prediction
pip install -r Requirements.txt
streamlit run app.py
```

---

## 🛠️ Technologies Used

**Language:** Python

**Libraries:** NumPy · Pandas · Scikit-learn · Matplotlib · SHAP · Streamlit

**Environment:** Google Colab · Visual Studio Code

---

## 📁 Repository Structure

```
Patient-Adherence-Prediction/
│
├── images/                          # App screenshots & evaluation plots
├── app.py                           # Streamlit application
├── train_model.ipynb                # Model training & evaluation notebook
├── patient_Adherence_prediction.pkl # Serialized preprocessing + model pipeline
├── patient_adherence_dataset.csv    # Dataset
├── Requirements.txt                 # Project dependencies
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Probability calibration (Platt scaling / isotonic regression)
- [ ] Model monitoring & drift detection in production
- [ ] Cross-validation comparison dashboard
- [ ] Interactive explainability dashboard
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Automated retraining pipeline

---

## 🎓 Key Learning Outcomes

- End-to-end supervised machine learning workflow
- Data preprocessing using Scikit-learn `Pipeline` and `ColumnTransformer`
- Hyperparameter tuning using `RandomizedSearchCV`
- Model evaluation using multiple classification metrics beyond accuracy
- Model explainability using SHAP for clinical stakeholder communication
- Machine learning model deployment using Streamlit

---

## 📬 Author

**Ala Sushanth**
GitHub: [@AlaSushanth](https://github.com/AlaSushanth)
