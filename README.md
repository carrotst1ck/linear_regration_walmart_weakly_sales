
````markdown
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
````

## Этап 1 Обучение модели

Файл `train.py`:

* загружает датасет
* обрабатывает данные
* создаёт новые признаки
* обучает модель Linear Regression
* считает метрики качества
* сохраняет модель в `model.pkl`
* сохраняет список колонок в `model_columns.pkl`

### Запуск

```bash
python train.py
```

### Что выводится

* `MAE` — средняя абсолютная ошибка
* `R2` — коэффициент детерминации
* `Accuracy` — точность модели в процентах

## Этап 2 Тестирование через терминал

Файл `test.py`:

* загружает сохранённую модель
* просит пользователя ввести значения
* делает предсказание
* выводит результат в терминале

### Запуск

```bash
python test.py
```

### Пример ввода

```text
Temperature: 25
Fuel_Price: 3.0
CPI: 200
Holiday_Flag: 0
Month: 6
Week: 20
Year: 2012
Prev_Week_Sales: 1500000
Prev_2_Week_Sales: 1450000
Store: 1
```

### Пример вывода

```text
Predicted Weekly_Sales: $1,523,456.78
```

## Этап 3 Веб-интерфейс

Файл `app.py` создаёт одностраничный сайт через Streamlit

На сайте есть:

* информация о модели
* метрики качества
* график Actual vs Predicted
* 10 строк из датасета
* объяснение preprocessing
* форма ввода данных
* кнопка `Predict`
* результат предсказания

### Запуск

```bash
streamlit run app.py
```

Если команда не работает:

```bash
python -m streamlit run app.py
```

## Установка библиотек

```bash
pip install pandas scikit-learn matplotlib streamlit joblib
```

Если используется виртуальное окружение `.venv`:

```bash
.\.venv\Scripts\python.exe -m pip install pandas scikit-learn matplotlib streamlit joblib
```

## Предобработка данных

В проекте были выполнены следующие шаги preprocessing:

* `Date` преобразована в:

  * `Month`
  * `Week`
  * `Year`
* для каждого магазина добавлены:

  * `Prev_Week_Sales`
  * `Prev_2_Week_Sales`
* `Store` преобразован в dummy columns
* `Weekly_Sales` используется как целевая переменная

## Почему были удалены или изменены некоторые колонки

* `Date` не использовалась напрямую, потому что модель лучше работает с отдельными временными признаками
* `Store` был преобразован в dummy columns, потому что линейная регрессия не должна воспринимать номер магазина как обычное число
* `Weekly_Sales` не используется как входной признак, потому что это то значение, которое модель должна предсказывать

## Датасет

Использован датасет Walmart с Kaggle

Источник:
`https://www.kaggle.com/datasets/yasserh/walmart-dataset`

## Результат

Проект показывает, как можно:

* обучить модель линейной регрессии
* сохранить модель через Joblib
* использовать модель в терминале
* подключить модель к веб-интерфейсу

## Автор

Проект выполнен в учебных целях

Если хочешь, я могу сразу сделать тебе ещё и **более красивую GitHub-версию README с эмодзи, разделителями и аккуратным оформлением**.
```
