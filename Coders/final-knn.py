import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare the data
data = pd.read_csv('Coders/edited-fruits.csv')
X = data[['color_score', 'width', 'height']]
y = data['fruit_label']

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

def plot_knn(X, y, k, feature1, feature2):
    # Create mesh grid
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Fit KNN and predict
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X[[feature1, feature2]], y)
    
    # Create a DataFrame for prediction
    mesh_df = pd.DataFrame({feature1: xx.ravel(), feature2: yy.ravel()})
    Z = knn.predict(mesh_df).reshape(xx.shape)
    
    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[feature1], X[feature2], c=y, alpha=0.8)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    
    # Calculate and return accuracy scores
    y_pred = knn.predict(X_test_scaled[[feature1, feature2]])
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy

# Plot setup with smaller figure size
fig, axs = plt.subplots(3, 3, figsize=(12, 9))  # Adjusted size for smaller plots
fig.suptitle('KNN Decision Boundaries and Accuracy Scores', fontsize=14)
feature_pairs = [('width', 'height'), ('color_score', 'width'), ('color_score', 'height')]
k_values = [1, 3, 5]

# Create plots and calculate accuracies
for i, (f1, f2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        plt.sca(axs[i, j])
        # Compute test accuracy
        test_acc = plot_knn(X_train_scaled, y_train, k, f1, f2)
        # Compute cross-validation accuracy
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)  # 5-fold cross-validation
        mean_cv_acc = np.mean(cv_scores)
        
        # Plot title with both accuracies
        plt.title(f'K={k}\nTest Acc: {test_acc*100:.2f}%\nCV Acc: {mean_cv_acc*100:.2f}%')

# Print accuracy scores
print("\nCross-Validation and Test Accuracy Scores:")
for f1, f2 in feature_pairs:
    print(f"Features: {f1}, {f2}")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)  # 5-fold cross-validation
        mean_cv_acc = np.mean(cv_scores)
        knn.fit(X_train_scaled[[f1, f2]], y_train)
        test_acc = accuracy_score(y_test, knn.predict(X_test_scaled[[f1, f2]]))
        print(f"K={k}: Mean CV Accuracy = {mean_cv_acc*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
