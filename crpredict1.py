import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import joblib
import plotly.graph_objects as go
import requests
import tempfile

# 默认数据文件路径
default_lab_data_file = "https://github.com/fushicorrosion/wl-cr-predict/raw/main/实验室数据-1021.1机器学习.xlsx"
default_weight_loss_file = "https://github.com/fushicorrosion/wl-cr-predict/raw/main/副本4.5-失重检查片腐蚀速率数据（总表删除交流空白）-1021.1机器学习.xlsx"
default_prediction_data_file = "https://github.com/fushicorrosion/wl-cr-predict/raw/main/用于预测用数据-1021.xlsx"
default_model_file = "https://github.com/fushicorrosion/wl-cr-predict/raw/main/trained_model.pkl"
# 设置默认保存路径
model_save_path = 'C:\\trained_model.pkl'

# 下载模型文件的函数
def download_model(url):
    response = requests.get(url)
    response.raise_for_status()  # 确保请求成功
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
        temp_file.write(response.content)
        return temp_file.name

# 创建侧边栏
st.sidebar.title("模型管理")

# 选择建模或预测模式，默认选择预测
mode = st.sidebar.radio("请选择模式", ("预测", "更新模型/建模"))


if mode == "更新模型/建模":
    st.sidebar.subheader("上传数据")
    lab_data_file = st.sidebar.file_uploader("上传实验室腐蚀速率数据", type=["xlsx"], key="lab_data") or default_lab_data_file
    weight_loss_file = st.sidebar.file_uploader("上传现场失重试片腐蚀速率数据", type=["xlsx"], key="weight_loss") or default_weight_loss_file
    prediction_data_file = st.sidebar.file_uploader("上传模拟现场工况数据", type=["xlsx"], key="prediction_data") or default_prediction_data_file

    # 点击建模按钮
    if st.sidebar.button("建模"):
        with st.spinner("请耐心等待建模完成..."):
            # 加载数据
            lab_data = pd.read_excel(lab_data_file)
            weight_loss_data = pd.read_excel(weight_loss_file)
            prediction_data = pd.read_excel(prediction_data_file)

            # 预处理数据
            X_lab = lab_data.drop(columns=['腐蚀速率/mm/a'])
            y_lab = lab_data['腐蚀速率/mm/a']
            X_weight_loss = weight_loss_data.drop(columns=['腐蚀速率/mm/a'])
            y_weight_loss = weight_loss_data['腐蚀速率/mm/a']

            # 确保两个数据集的列顺序一致
            X_lab = X_lab.reindex(sorted(X_lab.columns), axis=1)
            X_weight_loss = X_weight_loss.reindex(sorted(X_weight_loss.columns), axis=1)
            prediction_data = prediction_data.reindex(sorted(X_weight_loss.columns), axis=1)

            # 建立实验室数据模型
            model_lab = xgb.XGBRegressor(objective='reg:squarederror', alpha=0, colsample_bytree=0.7,
                                         learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, random_state=42)
            model_lab.fit(X_lab, y_lab)

            # 建立失重法数据模型
            model_weight = xgb.XGBRegressor(objective='reg:squarederror', alpha=0, colsample_bytree=0.7,
                                            learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, random_state=42)
            model_weight.fit(X_weight_loss, y_weight_loss)

            # 预测并创建新特征
            combined_X = pd.concat([X_weight_loss, X_lab, prediction_data], axis=0).reset_index(drop=True)
            y_lab_pred = model_lab.predict(combined_X)
            y_weight_loss_pred = model_weight.predict(combined_X)

            # 创建最终模型特征和标签，将'y_lab_pred'重命名为'0'
            y_lab_pred_series = pd.Series(y_lab_pred).rename('0').reset_index(drop=True)
            new_X = pd.concat([combined_X, y_lab_pred_series], axis=1)
            new_y = y_weight_loss_pred

            # 划分数据集并训练最终模型
            X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(new_X, new_y, test_size=0.2, random_state=42)
            final_model = xgb.XGBRegressor(objective='reg:squarederror', alpha=0, colsample_bytree=0.7,
                                           learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, random_state=42)
            final_model.fit(X_train_final, y_train_final)

            # 保存模型
            model_save_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl').name
            joblib.dump(final_model, model_save_path)

            # 显示下载按钮
            with open(model_save_path, "rb") as f:
                st.download_button(
                    label="下载模型文件",
                    data=f,
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )

            st.sidebar.success("模型已保存并可以下载。")

            # 显示训练和测试集 R2 分数
            y_final_train_pred = final_model.predict(X_train_final)
            y_final_test_pred = final_model.predict(X_test_final)
            st.write(f"训练集 R2 分数: {r2_score(y_train_final, y_final_train_pred):.4f}")
            st.write(f"测试集 R2 分数: {r2_score(y_test_final, y_final_test_pred):.4f}")

            # 可视化预测效果
            fig = go.Figure()

            # 添加训练集预测点
            fig.add_trace(go.Scatter(
                x=y_train_final,
                y=y_final_train_pred,
                mode='markers',
                name='训练集预测',
                marker=dict(color='blue', size=8, line=dict(width=1, color='black')),
                showlegend=True
            ))

            # 添加测试集预测点
            fig.add_trace(go.Scatter(
                x=y_test_final,
                y=y_final_test_pred,
                mode='markers',
                name='测试集预测',
                marker=dict(color='purple', size=8, line=dict(width=1, color='black')),
                showlegend=True
            ))

            # 添加 y=x 线
            fig.add_trace(go.Scatter(
                x=[min(y_test_final.min(), y_final_test_pred.min()), max(y_test_final.max(), y_final_test_pred.max())],
                y=[min(y_test_final.min(), y_final_test_pred.min()), max(y_test_final.max(), y_final_test_pred.max())],
                mode='lines',
                name='y = x',
                line=dict(color='red', dash='dash')
            ))

            # 更新布局
            fig.update_layout(
                title='最终模型预测效果',
                xaxis_title='实际腐蚀速率 (mm/a)',
                yaxis_title='预测腐蚀速率 (mm/a)',
                font=dict(family='Arial, sans-serif', size=12),
                width=800,
                height=600,
                legend=dict(x=0.1, y=0.9),
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig)

elif mode == "预测":
    st.sidebar.subheader("输入预测参数")
    input_data = {
        "断电电位最正值/VCSE": st.sidebar.number_input("断电电位最正值，VCSE", format="%.4f"),
        "断电电位最负值/VCSE": st.sidebar.number_input("断电电位最负值，VCSE", format="%.4f"),
        "断电电位平均值/VCSE": st.sidebar.number_input("断电电位平均值，VCSE", format="%.4f"),
        "断电电位正于阴极保护准则比例/%": st.sidebar.number_input("断电电位正于阴极保护准则比例，输入0~1", format="%.4f"),
        "断电电位正于阴极保护准则+50mV比例/%": st.sidebar.number_input("断电电位正于阴极保护准则+50mV比例，输入0~1", format="%.4f"),
        "断电电位正于阴极保护准则+100mV比例/%": st.sidebar.number_input("断电电位正于阴极保护准则+100mV比例，输入0~1", format="%.4f"),
        "交流电流密度最大值A/m2": st.sidebar.number_input("交流电流密度最大值，A/m2", format="%.4f"),
        "交流电流密度最小值A/m2": st.sidebar.number_input("交流电流密度最小值，A/m2", format="%.4f"),
        "交流电流密度平均值A/m2": st.sidebar.number_input("交流电流密度平均值，A/m2", format="%.4f"),
        "腐蚀速率/mm/a": st.sidebar.number_input("实验室腐蚀速率，mm/a", format="%.4f"),
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.rename(columns={"腐蚀速率/mm/a": "0"})  # 将列名改为'0'

    # 手动上传或选择模型文件
    model_file = st.sidebar.file_uploader("选择模型文件", type=["pkl"], key="model_file")

    # 加载模型
    if model_file is not None:
        final_model = joblib.load(model_file)
        st.sidebar.success("加载了选择的模型进行预测。")
    else:
        if os.path.exists(model_save_path):
            final_model = joblib.load(model_save_path)
            st.sidebar.success("加载了新建的模型进行预测。")
        else:
            st.sidebar.warning("新建模型不存在，使用默认模型。")
            model_save_path1 = download_model(default_model_file)  # 下载并获取模型路径
            final_model = joblib.load(model_save_path1)
            st.sidebar.success("加载了默认模型进行预测。")

    # 进行预测
    if st.sidebar.button("预测"):
        try:
            # 确保输入特征的顺序与训练时一致
            feature_names = ['交流电流密度平均值A/m2', '交流电流密度最大值A/m2', '交流电流密度最小值A/m2',
                             '断电电位平均值/VCSE', '断电电位最正值/VCSE', '断电电位最负值/VCSE',
                             '断电电位正于阴极保护准则+100mV比例/%', '断电电位正于阴极保护准则+50mV比例/%',
                             '断电电位正于阴极保护准则比例/%', '0']

            input_df = input_df.reindex(columns=feature_names)

            # 进行预测
            predicted_weight_loss = final_model.predict(input_df)[0]
            st.write(f"预测的现场失重试片腐蚀速率: {predicted_weight_loss:.4f} mm/a")

            # 可视化输入的实验室腐蚀速率和预测的现场失重试片腐蚀速率
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=["实验室腐蚀速率", "预测的现场失重试片腐蚀速率"],
                y=[input_data["腐蚀速率/mm/a"], predicted_weight_loss],
                marker_color=['blue', 'purple'],
                showlegend=False
            ))

            # 更新布局
            fig.update_layout(
                title='实验室与预测的现场失重试片腐蚀速率对比',
                yaxis_title='腐蚀速率 (mm/a)',
                font=dict(family='Arial, sans-serif', size=12),
                width=800,
                height=600,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")


