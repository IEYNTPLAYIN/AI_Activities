import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare the data
data = pd.read_csv('fruits.csv')
X = data[['color_score', 'width', 'height']]
y = data['fruit_label']
fruit_names = data['fruit_name'].unique()

# Define color map for fruits
fruit_colors = {1: 'red', 2: 'yellow', 3: 'orange', 4: 'pink'}

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

def plot_knn(ax, X, y, k, feature1, feature2, new_data=None):
    """
    Plot KNN decision boundaries and data points for given features.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on
    X (pd.DataFrame): The feature data
    y (pd.Series): The target labels
    k (int): The number of neighbors to use for KNN
    feature1, feature2 (str): The names of the two features to plot
    new_data (pd.DataFrame, optional): New data point to classify
    
    Returns:
    int or None: The predicted label for new_data, if provided
    """
    
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
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    ax.scatter(X[feature1], X[feature2], c=[fruit_colors[label] for label in y], alpha=0.8)
    
    prediction = None
    if new_data is not None:
        ax.scatter(new_data[feature1], new_data[feature2], color='green', marker='*', s=200)
        
        # Find k nearest neighbors
        distances, indices = knn.kneighbors(new_data[[feature1, feature2]])
        
        # Plot broken lines to nearest neighbors
        for i in range(k):
            ax.plot([new_data[feature1].values[0], X[feature1].iloc[indices[0][i]]],
                    [new_data[feature2].values[0], X[feature2].iloc[indices[0][i]]],
                    'k--', alpha=0.3)
            
        # Print the average distance to the nearest neighbors
        avg_distance = np.mean(distances)
        print(f'Average distance to the {k} nearest neighbors: {avg_distance}')
        
        prediction = knn.predict(new_data[[feature1, feature2]])[0]
    
    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    
    return prediction


# Calculate figure size to fit within 1080px vertical height
fig_width = 8  # Adjust this value to change the aspect ratio
fig_height = min(fig_width * 0.75, 10)  # Limit height to 1080px (assuming 108 DPI)

# Plot setup
fig, axs = plt.subplots(3, 3, figsize=(fig_width, fig_height))
fig.suptitle('KNN Decision Boundaries and Accuracy Scores', fontsize=14)
feature_pairs = [('width', 'height'), ('color_score', 'width'), ('color_score', 'height')]
k_values = [1, 3, 5]

# Create plots and calculate accuracies
for i, (f1, f2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        ax = axs[i, j]
        plot_knn(ax, X_train_scaled, y_train, k, f1, f2)
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Calculate cross-validation accuracy
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)
        mean_cv_acc = np.mean(cv_scores)
        
        # Calculate test accuracy
        knn.fit(X_train_scaled[[f1, f2]], y_train)
        y_pred = knn.predict(X_test_scaled[[f1, f2]])
        test_acc = accuracy_score(y_test, y_pred)
        
        ax.set_title(f'K={k}\nCV Acc: {mean_cv_acc*100:.2f}%\nTest Acc: {test_acc*100:.2f}%', fontsize=9)

plt.tight_layout()
plt.show()

# New data
new_data = pd.DataFrame([[0.82, 7.7, 7.9]], columns=['color_score', 'width', 'height'])
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

# Plot new data
fig, axs = plt.subplots(3, 3, figsize=(fig_width, fig_height))
fig.suptitle('KNN Decision Boundaries with New Data Point', fontsize=14)

for i, (f1, f2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        ax = axs[i, j]
        prediction = plot_knn(ax, X_train_scaled, y_train, k, f1, f2, new_data_scaled)
        fruit_name = fruit_names[prediction - 1]  # Assuming fruit labels start from 1
        ax.set_title(f'K={k}: {fruit_name}\n{f1}, {f2}', fontsize=10)

plt.tight_layout()
plt.show()

# Calculate and store accuracy scores
accuracy_data = []
for f1, f2 in feature_pairs:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Cross-validation accuracy
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)
        mean_cv_acc = np.mean(cv_scores)
        
        # Test accuracy
        knn.fit(X_train_scaled[[f1, f2]], y_train)
        y_pred = knn.predict(X_test_scaled[[f1, f2]])
        test_acc = accuracy_score(y_test, y_pred)
        
        accuracy_data.append({
            'Features': f'{f1}, {f2}',
            'K': k,
            'CV Accuracy': mean_cv_acc,
            'Test Accuracy': test_acc
        })

# Create a DataFrame from the accuracy data
accuracy_df = pd.DataFrame(accuracy_data)

# Create a bar plot to compare CV and Test accuracies
plt.figure(figsize=(15, 10))
bar_width = 0.35
index = np.arange(len(accuracy_df))

plt.bar(index, accuracy_df['CV Accuracy'], bar_width, label='CV Accuracy', alpha=0.8)
plt.bar(index + bar_width, accuracy_df['Test Accuracy'], bar_width, label='Test Accuracy', alpha=0.8)

plt.xlabel('Feature Pairs and K Values')
plt.ylabel('Accuracy')
plt.title('Comparison of Cross-Validation and Test Accuracies')
plt.xticks(index + bar_width / 2, [f"{row['Features']}, K={row['K']}" for _, row in accuracy_df.iterrows()], rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.show()

# Print accuracy scores
print("\nAccuracy Scores:")
for _, row in accuracy_df.iterrows():
    print(f"\nFeatures: {row['Features']}, K={row['K']}")
    print(f"CV Accuracy = {row['CV Accuracy']*100:.2f}%, Test Accuracy = {row['Test Accuracy']*100:.2f}%")