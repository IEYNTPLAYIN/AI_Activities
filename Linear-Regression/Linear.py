import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

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
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi ** 2 for xi in x)
    sum_y_squared = sum(yi ** 2 for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = math.sqrt((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2))
    
    if denominator == 0:
        return 0  # Avoid division by zero
    return numerator / denominator

# Function for linear regression
def linear_regression(x, y):
    x_mean, y_mean = mean(x), mean(y)
    sx, sy = standard_deviation(x), standard_deviation(y)
    r = pearson_correlation(x, y)
    
    # Slope and intercept
    slope = r * (sy / sx)
    intercept = y_mean - slope * x_mean
    return slope, intercept, r, sx, sy

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
slope, intercept, r, sx, sy = linear_regression(x_train, y_train)

# Compute residuals and predictions for the test set
y_test_pred, test_residuals, test_squared_residuals = compute_residuals(x_test, y_test, slope, intercept)

# Create a DataFrame for the testing table
test_df = pd.DataFrame({
    'Months': x_test,
    'Sales': y_test,
    'Predicted_Y': y_test_pred,
    'Squared_Residuals': test_squared_residuals
})

# Create a DataFrame for the regression statistics
stats_df = pd.DataFrame({
    'Parameters': ['Pearson Correlation', 'Standard Deviation of X', 'Standard Deviation of Y', 'Slope', 'Intercept'],
    'Value': [round(r, 4), round(sx, 2), round(sy, 2), round(slope, 4), round(intercept, 2)]
})


# Plot the overall results (trend line and actual data points)
plt.figure(figsize=(10, 6))
plt.scatter(df['Date'], y, label="Actual Sales", color="blue", alpha=0.6)
plt.plot(df['Date'], [slope * xi + intercept for xi in x], color='red', label="Trend Line")
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# ---- Create and format the Regression Statistics Table ----
fig1, ax1 = plt.subplots(figsize=(10, 6))  # First figure for regression statistics
ax1.axis('off')
ax1.axis('tight')

# Create the regression statistics table
stats_table = ax1.table(cellText=stats_df.values,
                        colLabels=stats_df.columns,
                        loc='center',
                        cellLoc='center',
                        colColours=['#f5f5f5', '#f5f5f5'])  # Header background color

# Apply formatting to the regression table
stats_table.auto_set_font_size(False)
stats_table.set_fontsize(10)  # Reduce the font size for rows
stats_table.auto_set_column_width(col=list(range(len(stats_df.columns))))

# Adjust row heights (larger cell height)
for i, key in enumerate(stats_table.get_celld().keys()):
    cell = stats_table.get_celld()[key]
    if key[0] == 0:  # Header row
        cell.set_fontsize(10)  # Smaller font size for header text
        cell.set_text_props(weight='semibold')  # Semi-bold
        cell.set_facecolor('#A9A9A9')  # Gray background
        cell.set_edgecolor('gray')  # White borders
        cell.set_text_props(color='white')  # White text color
        cell.set_height(0.12)
    else:  # Row cells
        cell.set_edgecolor('gray')  # Border color for non-header cells
        cell.set_height(0.12)  # Adjust row height (increase to make cells taller)

# Set title for the first figure
ax1.set_title("Required Values for Linear Regression", fontsize=16, fontweight='bold')

# Display the first table
plt.show()

# ---- Create and format the Testing Data Table ----
fig2, ax2 = plt.subplots(figsize=(10, 6))  # Second figure for testing data table
ax2.axis('off')
ax2.axis('tight')

# Create the testing data table
test_table_data = test_df[['Months', 'Sales', 'Predicted_Y', 'Squared_Residuals']].round(2)
test_table = ax2.table(cellText=test_table_data.values,
                       colLabels=test_table_data.columns,
                       loc='center',
                       cellLoc='center',
                       colColours=['#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5'])  # Header background color

# Apply formatting to the testing table
test_table.auto_set_font_size(False)
test_table.set_fontsize(7)
test_table.auto_set_column_width(col=list(range(len(test_table_data.columns))))

# Style the header row
for (i, j), cell in test_table.get_celld().items():
    if i == 0:  # Header row
        cell.set_fontsize(8)  # Smaller font size for header text
        cell.set_text_props(weight='semibold')  # Semi-bold
        cell.set_facecolor('#A9A9A9')  # Gray background
        cell.set_edgecolor('gray')  # White borders
        cell.set_text_props(color='white')  # White text color
        cell.set_height(0.04)
    else:
        cell.set_edgecolor('gray')
        cell.set_height(0.04)

# Set title for the second figure
ax2.set_title("Linear Regression Analysis Table (Test Dataset)", fontsize=16, fontweight='bold')

# Display the second table
plt.show()
