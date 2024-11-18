import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    squared_residuals = [res ** 2 for res in residuals]
    return y_pred, residuals, squared_residuals

# Main
filepath = "monthly-car-sales.csv"  # Replace with your dataset's path
df = load_data(filepath)

# Extract X and Y
x = df['Months'].values
y = df['Sales'].values

# Split into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Perform linear regression on the training set
slope, intercept = linear_regression(x_train, y_train)

# Compute residuals and predictions for both training and testing sets
y_train_pred, train_residuals, train_squared_residuals = compute_residuals(x_train, y_train, slope, intercept)
y_test_pred, test_residuals, test_squared_residuals = compute_residuals(x_test, y_test, slope, intercept)

# Add results to a DataFrame for the testing set
test_df = pd.DataFrame({'Months': x_test, 'Sales': y_test, 'Predicted_Y': y_test_pred, 'Squared_Residuals': test_squared_residuals})

# Plot the overall results (trend line and actual data points)
plt.figure(figsize=(10, 6))
plt.scatter(df['Date'], y, label="Actual Sales", color="blue", alpha=0.6)
plt.plot(df['Date'], [slope * xi + intercept for xi in x], color='red', label="Trend Line")
plt.title("Sales Trend Over Time (Training + Testing)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# Display the testing table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')  # Hide axes
ax.axis('tight')

test_table_data = test_df[['Months', 'Sales', 'Predicted_Y', 'Squared_Residuals']].round(2)
test_table = ax.table(cellText=test_table_data.values,
                      colLabels=test_table_data.columns,
                      loc='center',
                      cellLoc='center')

test_table.auto_set_font_size(False)
test_table.set_fontsize(10)
test_table.auto_set_column_width(col=list(range(len(test_table_data.columns))))
plt.title("Testing Data Table")
plt.show()
