# Patient Adherence Prediction Website

A Streamlit website for the trained Patient Adherence Prediction model.

## Files

- `app.py` — Streamlit web application
- `patient_Adherence_prediction_final.pkl` — trained model
- `requirements.txt` — required Python packages

## Run locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the website

```bash
python -m streamlit run app.py
```

The browser should open the Streamlit application automatically.

## Important

The supplied pickle was created using scikit-learn 1.6.1. Keep:

```text
scikit-learn==1.6.1
```

Do not change the categorical encoding in the website independently of the saved pipeline. The website sends raw patient values to the saved pipeline, which performs its own preprocessing.

This is an academic/project demonstration and is not a medical diagnostic tool.
