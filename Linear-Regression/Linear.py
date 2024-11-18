import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure 'Date' is datetime type with Year-Month format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
    # Convert date to numerical format (months since the first date)
    df['Months'] = (df['Date'].dt.year - df['Date'].dt.year.min()) * 12 + df['Date'].dt.month - df['Date'].dt.month.min()
    return df

# Function to calculate mean
def mean(values):
    return sum(values) / len(values)

# Function to calculate standard deviation
def standard_deviation(values):
    mean_val = mean(values)
    return np.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values))

# Function to calculate Pearson Correlation
def pearson_correlation(x, y):
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = np.sqrt(sum((xi - x_mean) ** 2 for xi in x) * sum((yi - y_mean) ** 2 for yi in y))
    return numerator / denominator

# Function for linear regression
def linear_regression(x, y):
    x_mean, y_mean = mean(x), mean(y)
    sx, sy = standard_deviation(x), standard_deviation(y)
    r = pearson_correlation(x, y)
    
    # Slope and intercept
    slope = r * (sy / sx)
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Function to compute residuals
def compute_residuals(x, y, slope, intercept):
    y_pred = [slope * xi + intercept for xi in x]
    residuals = [yi - ypi for yi, ypi in zip(y, y_pred)]
    return y_pred, residuals

# Main
filepath = "monthly-car-sales.csv"  # Replace with your dataset's path
df = load_data(filepath)

# Extracting X and Y
x = df['Months'].values
y = df['Sales'].values

# Perform linear regression
slope, intercept = linear_regression(x, y)
y_pred, residuals = compute_residuals(x, y, slope, intercept)

# Output results
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"Residuals: {residuals}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(df['Date'], y, label="Actual Sales", color="blue")
plt.plot(df['Date'], y_pred, color='red', label="Trend Line")
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()