import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('rain_data.csv')  # Replace with your dataset file

# Define independent variables (features) and dependent variable (target)
humidity = data[['Humidity']]  # Independent variable (Humidity)
rain = data['Rain']  # Dependent variable (Rain)

# Split the dataset into training and testing sets (80% for training, 20% for testing)
humidity_train, humidity_test, rain_train, rain_test = train_test_split(humidity, rain, test_size=0.2, random_state=42, shuffle=False)

# Train the logistic regression model using the training set
model = LogisticRegression()
model.fit(humidity_train, rain_train)

# Extract coefficients
intercept = model.intercept_[0]  # Intercept (Y Intercept Coefficient)
humidity_coefficient = model.coef_[0][0]  # Coefficient for 'Humidity' (Independent Variable Coefficient)

# Summation of Independent Variable and Dependent Variable(Testing Set)
sum_humidity_test = humidity_test.sum()
sum_rain_test = rain_test.sum()

# Generate predictions for the test set
logit_test = intercept + humidity_coefficient * humidity_test
probability_test = 1 / (1 + np.exp(-logit_test))

# Apply decision threshold of 0.5
threshold = 0.5

# Create a table for Humidity, Rain, Logistic Regression (pi), and Predictions
table_data = pd.DataFrame({
    'Humidity': humidity_test['Humidity'].values.flatten(),
    'Rain': rain_test,
    'Logistic Regression (pi)': probability_test.values.ravel()
})

print("\nTable for Humidity, Rain, Logistic Regression (pi):")
print(table_data.to_string(index=False))

# Generate predictions for plotting the regression curve
humidity_range = np.linspace(humidity.min().values[0], humidity.max().values[0], 300).reshape(-1, 1)
logit = intercept + humidity_coefficient * humidity_range  # Compute the logit function
probability = 1 / (1 + np.exp(-logit))  # Transform logit into probability

# Plot the actual data points and logistic regression curve using the test set (20% of the data)
plt.figure(figsize=(10, 6))
plt.scatter(humidity_test, rain_test, color='blue', label='Test Data (Rain)', zorder=3)
plt.plot(humidity_range, probability, color='orange', label='Logistic Regression Curve (π)', zorder=2)

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
                  colColours=['lightblue'] * 4,
                  bbox=[0.1, 0.1, 0.8, 0.8])  # Adjust bbox to position the table

# Show the table
plt.show()

# Create another figure for the computed values
plt.figure(figsize=(8, 2))
plt.axis('off')  # Hide the axes

# Add computed values to the table data
computed_values = pd.DataFrame({
    'Metric': ['Summation of Independent Variable (Testing Set)', 'Summation of Dependent Variable (Testing Set)', 'Y Intercept Coefficient', 'Independent Variable Coefficient'],
    'Value': [sum_humidity_test['Humidity'], sum_rain_test, intercept, humidity_coefficient]
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
print(f"Summation of Independent Variable for Testing Set: {sum_humidity_test}")
print(f"Summation of Dependent Variable for Testing Set: {sum_rain_test}")
print(f"Y Intercept Coefficient: {intercept}")
print(f"Independent Variable Coefficient: {humidity_coefficient}")