import joblib
import pandas as pd

try:
    model = joblib.load("model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except FileNotFoundError:
    print("Сначала запусти train.py")
    exit()

temperature = float(input("Temperature: "))
fuel_price = float(input("Fuel_Price: "))
cpi = float(input("CPI: "))
holiday_flag = int(input("Holiday_Flag (0 or 1): "))
month = int(input("Month: "))
week = int(input("Week: "))
year = int(input("Year: "))
prev_week_sales = float(input("Prev_Week_Sales: "))
prev_2_week_sales = float(input("Prev_2_Week_Sales: "))
store = int(input("Store: "))

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

store_col = f"Store_{store}"
if store_col in input_dict:
    input_dict[store_col] = 1

input_data = pd.DataFrame([input_dict])

prediction = model.predict(input_data)[0]

print(f"Predicted Weekly_Sales: ${prediction:,.2f}")