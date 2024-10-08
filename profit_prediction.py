import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("enter_csv_file")
# Split the data into features and target variable
X = data.drop('Profit', axis=1)
y = data['Profit']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred_LR = LR.predict(X_test)
mse_LR = mean_squared_error(y_test, y_pred_LR)
mae_LR = mean_absolute_error(y_test, y_pred_LR)
r2_LR = r2_score(y_test, y_pred_LR)

# Print evaluation metrics for Linear Regression
print("Linear Regression:")
print(f"MSE: {mse_LR}")
print(f"MAE: {mae_LR}")
print(f"R-squared: {r2_LR}")

# Scatter plot of actual vs predicted values for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_LR)
plt.title("Linear Regression - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual plot for Linear Regression
residuals_LR = y_test - y_pred_LR
plt.figure(figsize=(8, 6))
sns.residplot(y_pred_LR, residuals_LR, lowess=True, line_kws={'color': 'red'})
plt.title("Linear Regression - Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Train Decision Tree model
DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred_DT = DT.predict(X_test)
mse_DT = mean_squared_error(y_test, y_pred_DT)
mae_DT = mean_absolute_error(y_test, y_pred_DT)
r2_DT = r2_score(y_test, y_pred_DT)

# Print evaluation metrics for Decision Tree
print("\nDecision Tree:")
print(f"MSE: {mse_DT}")
print(f"MAE: {mae_DT}")
print(f"R-squared: {r2_DT}")

# Scatter plot of actual vs predicted values for Decision Tree
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_DT)
plt.title("Decision Tree - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual plot for Decision Tree
residuals_DT = y_test - y_pred_DT
plt.figure(figsize=(8, 6))
sns.residplot(y_pred_DT, residuals_DT, lowess=True, line_kws={'color': 'red'})
plt.title("Decision Tree - Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Train Random Forest model
RFR = RandomForestRegressor()
RFR.fit(X_train, y_train)
y_pred_RFR = RFR.predict(X_test)
mse_RFR = mean_squared_error(y_test, y_pred_RFR)
mae_RFR = mean_absolute_error(y_test, y_pred_RFR)
r2_RFR = r2_score(y_test, y_pred_RFR)

# Print evaluation metrics for Random Forest
print("\nRandom Forest:")
print(f"MSE: {mse_RFR}")
print(f"MAE: {mae_RFR}")
print(f"R-squared: {r2_RFR}")

# Scatter plot of actual vs predicted values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_RFR)
plt.title("Random Forest - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual plot for Random Forest
residuals_RFR = y_test - y_pred_RFR
plt.figure(figsize=(8, 6))
sns.residplot(y_pred_RFR, residuals_RFR, lowess=True, line_kws={'color': 
'red'})
plt.title("Random Forest - Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Train Support Vector Regression (SVR) model
SVR = SVR(kernel='linear')
SVR.fit(X_train, y_train)
y_pred_SVR = SVR.predict(X_test)
mse_SVR = mean_squared_error(y_test, y_pred_SVR)
mae_SVR = mean_absolute_error(y_test, y_pred_SVR)
r2_SVR = r2_score(y_test, y_pred_SVR)

# Print evaluation metrics for SVR
print("\nSupport Vector Regression (SVR):")
print(f"MSE: {mse_SVR}")
print(f"MAE: {mae_SVR}")
print(f"R-squared: {r2_SVR}")

# Scatter plot of actual vs predicted values for SVR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_SVR)
plt.title("Support Vector Regression (SVR) - Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Residual plot for SVR
residuals_SVR = y_test - y_pred_SVR
plt.figure(figsize=(8, 6))
sns.residplot(y_pred_SVR, residuals_SVR, lowess=True, line_kws={'color': 
'red'})
plt.title("Support Vector Regression (SVR) - Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

# Choose the best model based on mean squared error
models = {
"Linear Regression": mse_LR,
"Decision Tree": mse_DT,
"Random Forest": mse_RFR,
"SVR": mse_SVR
}
best_model = min(models, key=models.get)
print(f"Best Model: {best_model}")

# Create tkinter GUI application
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
root = tk.Tk()
root.title("Profit Prediction")
root.geometry("400x300")

# Get user input
def predict_profit():
   rd_spend = float(entry_rd_spend.get())
   admin_cost = float(entry_admin_cost.get())
   marketing_spend = float(entry_marketing_spend.get())

# Predict profit using the best model
if best_model == "Linear Regression":
     profit = LR.predict([[rd_spend, admin_cost, marketing_spend]])
elif best_model == "Decision Tree":
    profit = DT.predict([[rd_spend, admin_cost, marketing_spend]])
elif best_model == "Random Forest":
    profit = RFR.predict([[rd_spend, admin_cost, marketing_spend]])
elif best_model == "SVR":
    profit = SVR.predict([[rd_spend, admin_cost, marketing_spend]])

# Display the predicted profit
predicted_profit_label.config(text=f"Predicted Profit: ${profit[0]:.2f}")


# Create GUI elements
label_rd_spend = ttk.Label(root, text="R&D Spend:")
label_rd_spend.pack()
entry_rd_spend = ttk.Entry(root)
entry_rd_spend.pack()

label_admin_cost = ttk.Label(root, text="Administration Cost:")
label_admin_cost.pack()
entry_admin_cost = ttk.Entry(root)
entry_admin_cost.pack()

label_marketing_spend = ttk.Label(root, text="Marketing Spend:")
label_marketing_spend.pack()
entry_marketing_spend = ttk.Entry(root)
entry_marketing_spend.pack()

predict_button = ttk.Button(root, text="Predict Profit", command=predict_profit)
predict_button.pack()
predicted_profit_label = ttk.Label(root, text="")
predicted_profit_label.pack()

root.mainloop()
