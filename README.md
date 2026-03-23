# Walmart Weekly Sales Predictor

Это простой ML-проект для прогнозирования недельных продаж Walmart с помощью **Linear Regression**

Проект состоит из 3 этапов:

1. **Обучение модели**
2. **Тестирование модели через терминал**
3. **Веб-интерфейс через Streamlit**

## О проекте

Модель предсказывает `Weekly_Sales` на основе данных из датасета Walmart

Для обучения используются такие признаки:

- Temperature
- Fuel_Price
- CPI
- Holiday_Flag
- Month
- Week
- Year
- Prev_Week_Sales
- Prev_2_Week_Sales
- Store

## Использованные технологии

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## Структура проекта

```bash
PythonProject4/
│
├── Walmart.csv
├── train.py
├── test.py
├── app.py
├── model.pkl
├── model_columns.pkl
└── README.md
