# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Coders/edited-fruits.csv')

# Split features (X) and labels (y)
X = data[['width', 'height']]  # Ensure feature names are consistent
y = data['fruit_label']  # Output labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Ensure X_test is a DataFrame with the same column names as X_train
X_test_df = pd.DataFrame(X_test, columns=['width', 'height'])

# Make predictions
y_pred = knn.predict(X_test_df)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plotting the decision boundary
h = 0.02  # Step size in the mesh
x_min, x_max = X['width'].min() - 1, X['width'].max() + 1
y_min, y_max = X['height'].min() - 1, X['height'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on grid points
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and training points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot training points
sns.scatterplot(x='width', y='height', hue='fruit_label', data=data, 
                palette='coolwarm', s=100, edgecolor='k')

plt.title(f'KNN Decision Boundary (k=3, Accuracy={accuracy * 100:.2f}%)')
plt.xlabel('width')
plt.ylabel('height')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
