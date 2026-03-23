import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.4f}")
print(f"Accuracy: {r2 * 100:.2f}%")

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("model.pkl создан")
print("model_columns.pkl создан")