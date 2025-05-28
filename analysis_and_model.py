import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


def analysis_and_model_page():
    st.title("Анализ данных и прогнозирование отказов оборудования")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Переименование столбцов для исключения ошибок в XGBoost
        data = data.rename(columns={
            'Air temperature [K]': 'Air_temperature_K',
            'Process temperature [K]': 'Process_temperature_K',
            'Rotational speed [rpm]': 'Rotational_speed_rpm',
            'Torque [Nm]': 'Torque_Nm',
            'Tool wear [min]': 'Tool_wear_min'
        })

        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

        # Масштабирование признаков
        numerical_features = [
            'Air_temperature_K',
            'Process_temperature_K',
            'Rotational_speed_rpm',
            'Torque_Nm',
            'Tool_wear_min'
        ]
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        st.session_state.scaler = scaler

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Инициализация моделей
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "SVM": SVC(kernel='linear', probability=True, random_state=42)
        }

        # Обучение и оценка моделей
        metrics = []
        best_model_info = {'name': '', 'score': -1, 'model': None}

        st.subheader("Анализ моделей")
        with st.expander("Показать детализацию моделей"):
            # Создаем табы для каждой модели
            tabs = st.tabs(list(models.keys()))

            for (name, model), tab in zip(models.items(), tabs):
                with tab:
                    # Обучение модели
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Расчет метрик
                    accuracy = accuracy_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)

                    # Визуализация Confusion Matrix
                    st.subheader("Матрица ошибок")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Предсказанные')
                    ax.set_ylabel('Истинные')
                    st.pyplot(fig)

                    # Отчет классификации в таблице
                    st.subheader("Отчет классификации")
                    df_report = pd.DataFrame(report).transpose().reset_index()
                    df_report.columns = ['Класс', 'Precision', 'Recall', 'F1-Score', 'Поддержка']
                    st.dataframe(
                        df_report.style.format({'Precision': '{:.2f}', 'Recall': '{:.2f}', 'F1-Score': '{:.2f}'})
                        .set_table_styles([{
                            'selector': 'td',
                            'props': [('border', '1px solid grey !important')]
                        }]),
                        #height=300
                        height=250
                    )

                    # Расчет комплексной оценки
                    score = (accuracy * 0.3 + roc_auc * 0.4 +
                             report['1']['f1-score'] * 0.3)  # Весовая формула

                    # Обновление лучшей модели
                    if score > best_model_info['score']:
                        best_model_info.update({
                            'name': name,
                            'score': score,
                            'model': model
                        })

                    # Сохранение метрик
                    metrics.append({
                        "Модель": name,
                        "Accuracy": accuracy,
                        "ROC-AUC": roc_auc,
                        "F1-Score (1)": report['1']['f1-score']
                    })

            # Вывод сводной таблицы
            st.subheader("Сравнение моделей")
            df_summary = pd.DataFrame(metrics)
            st.dataframe(
                df_summary.style.format({
                    'Accuracy': '{:.2%}',
                    'ROC-AUC': '{:.2%}',
                    'F1-Score (1)': '{:.2%}'
                }).highlight_max(color='lightgreen')
            )

        # Сохранение лучшей модели
        st.session_state.best_model = best_model_info['model']
        st.success(f"""
        **Оптимальная модель**: {best_model_info['name']}
        - Комплексная оценка: {best_model_info['score']:.2f}
        - Accuracy: {df_summary[df_summary['Модель'] == best_model_info['name']]['Accuracy'].values[0]:.2%}
        - ROC-AUC: {df_summary[df_summary['Модель'] == best_model_info['name']]['ROC-AUC'].values[0]:.2%}
        """)

        # Интерфейс прогнозирования
        st.subheader("Прогнозирование отказа")
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                product_type = st.selectbox("Тип продукта", ['L', 'M', 'H'])
                air_temp = st.number_input("Температура воздуха [K]", 300.0)
                process_temp = st.number_input("Температура процесса [K]", 310.0)
            with col2:
                rotational_speed = st.number_input("Скорость вращения [rpm]", 1500)
                torque = st.number_input("Крутящий момент [Nm]", 40.0)
                tool_wear = st.number_input("Износ инструмента [min]", 100)

            if st.form_submit_button("Рассчитать"):
                input_data = pd.DataFrame({
                    'Type': [0 if product_type == 'L' else 1 if product_type == 'M' else 2],
                    'Air_temperature_K': [air_temp],
                    'Process_temperature_K': [process_temp],
                    'Rotational_speed_rpm': [rotational_speed],
                    'Torque_Nm': [torque],
                    'Tool_wear_min': [tool_wear]
                })

                # Масштабирование
                input_data[numerical_features] = st.session_state.scaler.transform(input_data[numerical_features])

                # Прогноз
                if 'best_model' in st.session_state:
                    model = st.session_state.best_model
                    proba = model.predict_proba(input_data)[0][1]
                    prediction = model.predict(input_data)[0]

                    # Визуализация результатов
                    st.metric(label="Вероятность отказа", value=f"{proba:.2%}")
                    st.metric(label="Прогноз",
                              value="🚨 Отказ оборудования" if prediction == 1 else "✅ Отказа нет")

# Стилизация
st.markdown("""
<style>
div[data-testid="stExpander"] div[role="button"] p {
    font-size: 1.2rem !important;
    font-weight: bold !important;
}
div.stDataFrame div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)