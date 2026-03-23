import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Walmart Weekly Sales Predictor", layout="wide")

st.title("Walmart Weekly Sales Predictor")

data = pd.read_csv("Walmart.csv")
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data["Month"] = data["Date"].dt.month
data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
data["Year"] = data["Date"].dt.year
data["Prev_Week_Sales"] = data.groupby("Store")["Weekly_Sales"].shift(1)
data["Prev_2_Week_Sales"] = data.groupby("Store")["Weekly_Sales"].shift(2)
data = data.dropna().copy()

store_dummies = pd.get_dummies(data["Store"], prefix="Store", drop_first=True)

X = pd.concat(
    [
        data[["Temperature", "Fuel_Price", "CPI", "Holiday_Flag", "Month", "Week", "Year", "Prev_Week_Sales", "Prev_2_Week_Sales"]],
        store_dummies
    ],
    axis=1
)

y = data["Weekly_Sales"]

model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

tab1, tab2, tab3, tab4 = st.tabs(["Model Info", "Graph", "Dataset", "Predict"])

with tab1:
    st.subheader("Model Accuracy and Error Metrics")

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:,.2f}")
    col2.metric("R²", f"{r2:.4f}")

    st.subheader("Preprocessing Explanation")
    st.write("The final model was trained using these features:")
    st.write("- Temperature")
    st.write("- Fuel_Price")
    st.write("- CPI")
    st.write("- Holiday_Flag")
    st.write("- Month")
    st.write("- Week")
    st.write("- Year")
    st.write("- Prev_Week_Sales")
    st.write("- Prev_2_Week_Sales")
    st.write("- Store as dummy columns")

    st.write("Removed or transformed columns:")
    st.write("- Date: converted into Month, Week, and Year")
    st.write("- Store: converted into dummy columns for linear regression")
    st.write("- Weekly_Sales: used as the target column, not as input")

    st.subheader("Final Dataset Used for Training")
    st.dataframe(X.head(10), use_container_width=True)

with tab2:
    st.subheader("Graph from Stage 1")

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, label="Predictions")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Ideal"
    )
    ax.set_xlabel("Actual Weekly Sales")
    ax.set_ylabel("Predicted Weekly Sales")
    ax.set_title("Actual vs Predicted Weekly Sales")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    st.pyplot(fig)

with tab3:
    st.subheader("10 Samples from the Original Dataset")
    st.dataframe(data.head(10), use_container_width=True)

with tab4:
    st.subheader("Enter Values for Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input("Temperature", value=25.0)
        fuel_price = st.number_input("Fuel_Price", value=3.0)
        cpi = st.number_input("CPI", value=200.0)

    with col2:
        holiday_flag = st.selectbox("Holiday_Flag", [0, 1])
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
        week = st.number_input("Week", min_value=1, max_value=53, value=20)

    with col3:
        year = st.number_input("Year", min_value=2010, max_value=2030, value=2012)
        prev_week_sales = st.number_input("Prev_Week_Sales", value=1500000.0)
        prev_2_week_sales = st.number_input("Prev_2_Week_Sales", value=1450000.0)

    store = st.number_input("Store", min_value=1, max_value=45, value=1)

    if st.button("Predict"):
        input_dict = {col: 0 for col in model_columns}

        input_dict["Temperature"] = temperature
        input_dict["Fuel_Price"] = fuel_price
        input_dict["CPI"] = cpi
        input_dict["Holiday_Flag"] = holiday_flag
        input_dict["Month"] = month
        input_dict["Week"] = week
        input_dict["Year"] = year
        input_dict["Prev_Week_Sales"] = prev_week_sales
        input_dict["Prev_2_Week_Sales"] = prev_2_week_sales

        store_col = f"Store_{int(store)}"
        if store_col in input_dict:
            input_dict[store_col] = 1

        input_data = pd.DataFrame([input_dict])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")