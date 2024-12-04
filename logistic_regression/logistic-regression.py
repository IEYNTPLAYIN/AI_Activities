import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('rain_data.csv')  # Replace with your dataset file

# Define independent variables (features) and dependent variable (target)
X = data[['Humidity']]  # Independent variable (Humidity)
y = data['Rain']  # Dependent variable (Rain)


# Encode the 'rain' column if necessary
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# Check the unique values in the target variable
unique_classes = np.unique(y)
print(f'Unique classes in the target variable: {unique_classes}')

# Split the dataset into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train the logistic regression model using the training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Extract coefficients (β₀ and β₁)
beta_0 = model.intercept_[0]  # Intercept (Y Intercept Coefficient)
beta_1 = model.coef_[0][0]    # Coefficient for 'Humidity' (Independent Variable Coefficient)

# Summation of Independent Variable
sum_X_train = X_train.sum()

# Summation of Dependent Variable
sum_y_train = y_train.sum()

# Generate predictions for the test set
logit_test = beta_0 + beta_1 * X_test
pi_test = 1 / (1 + np.exp(-logit_test))

# Create a table for Humidity, Rain, and Logistic Regression (pi)
table_data = pd.DataFrame({
    'Humidity': X_test['Humidity'].values.flatten(),
    'Rain': y_test,
    'Logistic Regression (pi)': pi_test.values.ravel()
})

print("\nTable for Humidity, Rain, and Logistic Regression (pi):")
print(table_data.to_string(index=False))

# Generate predictions for plotting the regression curve
X_range = np.linspace(X.min().values[0], X.max().values[0], 300).reshape(-1, 1)
logit = beta_0 + beta_1 * X_range  # Compute the logit function
pi = 1 / (1 + np.exp(-logit))      # Transform logit into probability

# Plot the actual data points and logistic regression curve using the test set (20% of the data)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data (Rain)', zorder=3)
plt.plot(X_range, pi, color='orange', label='Logistic Regression Curve (π)', zorder=2)

# Customize the plot
plt.title('Logistic Regression: Rain Prediction vs. Humidity (Test Data)', fontsize=14)
plt.xlabel('Humidity', fontsize=12)
plt.ylabel('Rain Probability', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()

# Create a new figure for the table
plt.figure(figsize=(8, 4))
plt.axis('off')  # Hide the axes

# Add the table to the new figure
table_data_rounded = table_data.round(3)  # Round the table data for better readability
table = plt.table(cellText=table_data_rounded.values,
                  colLabels=table_data_rounded.columns,
                  cellLoc='center',
                  loc='center',
                  colColours=['lightblue'] * 3,
                  bbox=[0.1, 0.1, 0.8, 0.8])  # Adjust bbox to position the table

# Show the table
plt.show()

# Create another figure for the computed values
plt.figure(figsize=(8, 2))
plt.axis('off')  # Hide the axes

# Add computed values to the table data
computed_values = pd.DataFrame({
    'Metric': ['Summation of Independent Variable', 'Summation of Dependent Variable', 'Y Intercept Coefficienct', 'Independent Variable Coefficient'],
    'Value': [sum_X_train['Humidity'], sum_y_train, beta_0, beta_1]
})

# Add the computed values table to the new figure
table_computed = plt.table(cellText=computed_values.values,
                           colLabels=computed_values.columns,
                           cellLoc='center',
                           loc='center',
                           colColours=['lightgreen', 'lightgreen'],
                           bbox=[0.1, 0.1, 0.8, 0.8])  # Adjust bbox to position the table

# Show the computed values table
plt.show()

# Display results
print(f"Summation of Independent Variable for Training Set: {sum_X_train}")
print(f"Summation of Dependent Variable for Training Set: {sum_y_train}")
print(f"Y Intercept Coefficient: {beta_0}")
print(f"Independent Variable Coefficient: {beta_1}")