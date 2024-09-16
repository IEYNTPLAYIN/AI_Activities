import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare the data
data = pd.read_csv('edited-fruits.csv')
X = data[['mass', 'width', 'height']]  # Features
y = data['fruit_label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Function to plot decision boundaries
def plot_decision_boundaries(X, y, k, feature1, feature2):
    # Create mesh grid
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Fit KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[[feature1, feature2]], y)
    
    # Predict on mesh grid
    mesh_points = pd.DataFrame({feature1: xx.ravel(), feature2: yy.ravel()})
    Z = knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[feature1], X[feature2], c=y, alpha=0.8)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'KNN (k={k}) Decision Boundary')

# Create subplots
fig, axs = plt.subplots(3, 3, figsize=(6.5, 10))
fig.suptitle('KNN Decision Boundaries for Different Feature Combinations and K Values', fontsize=10)

feature_pairs = [('width', 'height'), ('mass', 'width'), ('mass', 'height')]
k_values = [1, 3, 5]

for i, (feature1, feature2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        plt.sca(axs[i, j])
        plot_decision_boundaries(X_train_scaled, y_train, k, feature1, feature2)
        
        # Calculate accuracy
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled[[feature1, feature2]], y_train)
        y_pred = knn.predict(X_test_scaled[[feature1, feature2]])
        accuracy = accuracy_score(y_test, y_pred)
        
        plt.title(f'K={k}, {feature1} vs {feature2}\nAccuracy: {accuracy:.4f}')

plt.tight_layout()
plt.show()

# Print all accuracy scores
print("\nAccuracy Scores:")
for feature1, feature2 in feature_pairs:
    print(f"\nFeatures: {feature1}, {feature2}")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled[[feature1, feature2]], y_train)
        y_pred = knn.predict(X_test_scaled[[feature1, feature2]])
        accuracy = accuracy_score(y_test, y_pred)
        print(f"K={k}: {accuracy:.4f}")