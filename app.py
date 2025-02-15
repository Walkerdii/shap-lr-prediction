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
# 1️⃣ 加载数据和模型
# ==============================

# 使用 GitHub 上的文件
train_data_url = "https://raw.githubusercontent.com/Walkerdii/shap-lr-prediction/main/trainData_6.xlsx"
validation_data_url = "https://raw.githubusercontent.com/Walkerdii/shap-lr-prediction/main/valData_6.xlsx"

# 加载数据
df_train = pd.read_excel(train_data_url)
df_val = pd.read_excel(validation_data_url)

# 定义变量
target = "Prognosis"
continuous_vars = ['AGE', 'Number_of_diagnoses']
binary_vars = ["Excision_of_lesion_in_spinal_canal", "Lumbar_puncture"]
multiclass_vars = ['Patient_source', 'Timing_of_surgery']
all_features = continuous_vars + binary_vars + multiclass_vars  # 合并所有特征

# 变量映射
timing_mapping = {1: "Non-surgery", 2: "Surgery within 48h", 3: "Surgery over 48h"}
reverse_timing_mapping = {1: 0, 2: 1, 3: 2}  # 显示和计算的转换（从1,2,3回到0,1,2）

patient_mapping = {1: "Erban core", 2: "Erban fringe", 3: "County", 4: "Countryside"}
binary_mapping = {0: "No", 1: "Yes"}

# ==============================
# 2️⃣ 数据预处理（标准化）
# ==============================

# 标准化连续变量
scaler = StandardScaler()
df_train[continuous_vars] = scaler.fit_transform(df_train[continuous_vars])
df_val[continuous_vars] = scaler.transform(df_val[continuous_vars])  # 验证集应用训练集的标准化

# 分离 X 和 y
X_train = df_train[all_features]
y_train = df_train[target]
X_val = df_val[all_features]  # 仅用于 SHAP 解释

# ==============================
# 3️⃣ 定义并调参 Logistic 回归模型
# ==============================

pipe = Pipeline([
    ('lr', LogisticRegression(max_iter=5000))
])

# 定义超参数网格
param_grid = {
    'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'lr__penalty': ['l1', 'l2'],
    'lr__solver': ['liblinear', 'saga']
}

# 网格搜索 + 交叉验证
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=10, scoring='roc_auc',
                           verbose=1, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_

# 微调参数
fine_tuned_params = {
    'lr__C': [best_params['lr__C'] / 2, best_params['lr__C'], best_params['lr__C'] * 2],
    'lr__penalty': [best_params['lr__penalty']],
    'lr__solver': [best_params['lr__solver']]
}

# 微调网格搜索
fine_tuned_grid_search = GridSearchCV(pipe, param_grid=fine_tuned_params, cv=10, scoring='roc_auc',
                                      verbose=1, n_jobs=-1, refit=True)
fine_tuned_grid_search.fit(X_train, y_train)

# 获取最终模型
best_model = fine_tuned_grid_search.best_estimator_

# 训练最终模型
best_model.fit(X_train, y_train)

# ==============================
# 4️⃣ 计算 SHAP 解释
# ==============================

# 计算 SHAP 值
explainer = shap.Explainer(best_model['lr'], X_train)
shap_values = explainer(X_val)

# ==============================
# 5️⃣ Streamlit 网页界面
# ==============================

st.title("患者预后预测与解释")

# 用户输入
with st.form("prediction_form"):
    st.subheader("输入患者信息")

    # 连续变量
    age = st.number_input("AGE", min_value=0, max_value=120, value=50)
    num_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=20, value=5)

    # 二分类变量
    excision = st.selectbox("Excision of lesion in spinal canal", options=[0, 1], format_func=lambda x: binary_mapping[x])
    lumbar = st.selectbox("Lumbar puncture", options=[0, 1], format_func=lambda x: binary_mapping[x])

    # 多分类变量
    patient_source = st.selectbox("Patient Source", options=list(patient_mapping.keys()), format_func=lambda x: patient_mapping[x])
    timing_surgery = st.selectbox("Timing of Surgery", options=list(timing_mapping.keys()), format_func=lambda x: timing_mapping[x])

    submitted = st.form_submit_button("预测预后")

# ==============================
# 6️⃣ 计算预测结果与 SHAP 解释
# ==============================

if submitted:
    # 构造输入数据
    input_data = pd.DataFrame([[age, num_diagnoses, excision, lumbar, patient_source, timing_surgery]],
                              columns=all_features)

    # 进行标准化（只对连续变量）
    input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

    # 将 Timing_of_surgery 转换为模型计算所需的 0,1,2 格式
    input_data['Timing_of_surgery'] = input_data['Timing_of_surgery'].map(reverse_timing_mapping)

    # 进行预测
    prediction_prob = best_model.predict_proba(input_data)[0, 1]
    st.markdown(f"### 预测结果: **{prediction_prob:.2%}**  (预测为'预后不良'的概率)")

    # 计算 SHAP 值
    shap_values_single = explainer(input_data)

    # 显示 SHAP 力图
    st.subheader("解释力图")
    st_shap(shap.force_plot(explainer.expected_value, shap_values_single.values, input_data))
