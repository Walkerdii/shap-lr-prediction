import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from streamlit_shap import st_shap

# ==============================
# 1️⃣  加载数据和模型
# ==============================

# 训练数据路径
train_data_path = "D:/文档/Dr.wu/组内互助/翟/硬脊膜外血肿预后/5.模型验证/3.6变量/trainData_6.xlsx"
df_train = pd.read_excel(train_data_path)

# 定义变量
target = "Prognosis"
continuous_vars = ['AGE', 'Number_of_diagnoses']
binary_vars = ["Excision_of_lesion_in_spinal_canal", "Lumbar_puncture"]
multiclass_vars = ['Patient_source', 'Timing_of_surgery']
all_features = continuous_vars + binary_vars + multiclass_vars

# 变量映射
timing_mapping = {1: "Non-surgery", 2: "Surgery within 48h", 3: "Surgery over 48h"}
patient_mapping = {1: "Erban core", 2: "Erban fringe", 3: "County", 4: "Countryside"}
binary_mapping = {0: "No", 1: "Yes"}

# 数据预处理
scaler = StandardScaler()
df_train[continuous_vars] = scaler.fit_transform(df_train[continuous_vars])
X_train = df_train[all_features]
y_train = df_train[target]

# 训练 Logistic 回归模型
model = LogisticRegression(max_iter=5000, solver='liblinear', penalty='l1', C=1.0)
model.fit(X_train, y_train)

# 计算 SHAP 解释器
explainer = shap.Explainer(model, X_train)

# ==============================
# 2️⃣  Streamlit 网页界面
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
# 3️⃣  计算预测结果与 SHAP 解释
# ==============================

if submitted:
    # 构造输入数据
    input_data = pd.DataFrame([[age, num_diagnoses, excision, lumbar, patient_source, timing_surgery]],
                              columns=all_features)

    # 进行标准化（只对连续变量）
    input_data[continuous_vars] = scaler.transform(input_data[continuous_vars])

    # 进行预测
    prediction_prob = model.predict_proba(input_data)[0, 1]
    st.markdown(f"### 预测结果: **{prediction_prob:.2%}**  (预测为'预后不良'的概率)")

    # 计算 SHAP 值
    shap_values = explainer(input_data)

    # 显示 SHAP 力图
    st.subheader("解释力图")
    st_shap(shap.force_plot(explainer.expected_value, shap_values.values, input_data))

