import os
import pandas as pd
import shap
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from streamlit_shap import st_shap

# ==============================
# 1️⃣ Load Data and Model
# ==============================

# Using files from GitHub
train_data_url = "https://raw.githubusercontent.com/Walkerdii/shap-lr-prediction/main/trainData_6.xlsx"
validation_data_url = "https://raw.githubusercontent.com/Walkerdii/shap-lr-prediction/main/valData_6.xlsx"

# Load data
df_train = pd.read_excel(train_data_url)
df_val = pd.read_excel(validation_data_url)

# Define variables
target = "Prognosis"
continuous_vars = ['AGE', 'Number_of_diagnoses']
binary_vars = ["Excision_of_lesion_in_spinal_canal", "Lumbar_puncture"]
multiclass_vars = ['Patient_source', 'Timing_of_surgery']
all_features = continuous_vars + binary_vars + multiclass_vars  # Combine all features

# Variable mappings
timing_mapping = {1: "Non-surgery", 2: "Surgery within 48h", 3: "Surgery over 48h"}
reverse_timing_mapping = {1: 0, 2: 1, 3: 2}  # Mapping for display and calculation

patient_mapping = {1: "Erban core", 2: "Erban fringe", 3: "County", 4: "Countryside"}
binary_mapping = {0: "No", 1: "Yes"}

# ==============================
# 2️⃣ Data Preprocessing (Standardization)
# ==============================

# Standardize continuous variables
scaler = StandardScaler()
df_train[continuous_vars] = scaler.fit_transform(df_train[continuous_vars])
df_val[continuous_vars] = scaler.transform(df_val[continuous_vars])  # Apply the training set standardization to the validation set

# Split X and y
X_train = df_train[all_features]
y_train = df_train[target]
X_val = df_val[all_features]  # Only used for SHAP explanation

# ==============================
# 3️⃣ Define and Tune Logistic Regression Model
# ==============================

pipe = Pipeline([('lr', LogisticRegression(max_iter=5000))])

# Define hyperparameter grid
param_grid = {
    'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear', 'saga']
}

# Grid search + cross-validation
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=10, scoring='roc_auc',
                           verbose=1, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_

# Fine-tuning parameters
fine_tuned_params = {
    'lr__C': [best_params['lr__C'] / 2, best_params['lr__C'], best_params['lr__C'] * 2],
    'lr__penalty': [best_params['lr__penalty']],
    'lr__solver': [best_params['lr__solver']]
}

# Fine-tuned grid search
fine_tuned_grid_search = GridSearchCV(pipe, param_grid=fine_tuned_params, cv=10, scoring='roc_auc',
                                      verbose=1, n_jobs=-1, refit=True)
fine_tuned_grid_search.fit(X_train, y_train)

# Get the final model
best_model = fine_tuned_grid_search.best_estimator_

# Train the final model
best_model.fit(X_train, y_train)

# ==============================
# 4️⃣ Calculate SHAP Explanation
# ==============================

# Compute SHAP values
explainer = shap.Explainer(best_model['lr'], X_train)
shap_values = explainer(X_val)

# ==============================
# 5️⃣ Streamlit Web Interface
# ==============================

st.title("Final LR Prediction Model for Predicting Prognosis of Patients with Epidural Hematoma")

# User input
with st.form("prediction_form"):
    st.subheader("Input Patient Information")

    # Continuous variables
    age = st.number_input("AGE", min_value=0, max_value=120, value=50)
    num_diagnoses = st.number_input("Number of Diagnoses", value=5)

    # Binary variables
    excision = st.selectbox("Excision of lesion in spinal canal", options=[0, 1], format_func=lambda x: binary_mapping[x])
    lumbar = st.selectbox("Lumbar puncture", options=[0, 1], format_func=lambda x: binary_mapping[x])

    # Multiclass variables
    patient_source = st.selectbox("Patient Source", options=list(patient_mapping.keys()), format_func=lambda x: patient_mapping[x])
    timing_surgery = st.selectbox("Timing of Surgery", options=list(timing_mapping.keys()), format_func=lambda x: timing_mapping[x])

    submitted = st.form_submit_button("Predict")

# ==============================
# 6️⃣ Calculate Prediction Result and SHAP Explanation
# ==============================

if submitted:
    # Construct input data
    input_data = pd.DataFrame([[age, num_diagnoses, excision, lumbar, patient_source, timing_surgery]],
                              columns=all_features)

    # Standardize (only for continuous variables)
    input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

    # Convert Timing_of_surgery to the required format for model calculation
    input_data['Timing_of_surgery'] = input_data['Timing_of_surgery'].map(reverse_timing_mapping)

    # Make prediction
    prediction_prob = best_model.predict_proba(input_data)[0, 1]
    st.markdown(f"### Prediction Result: **{prediction_prob:.2%}** (Probability of 'Poor Prognosis' based on feature value)")

    # Calculate SHAP values
    shap_values_single = explainer(input_data)

    # Display SHAP force plot
    st.subheader("SHAP Force Plot")
    st_shap(shap.force_plot(explainer.expected_value, shap_values_single.values, input_data))
