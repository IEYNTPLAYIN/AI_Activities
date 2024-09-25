import pandas as pd # Used for data manipulation and analysis, providing data structures like DataFrame
import numpy as np # Used for numerical operations, providing support for arrays and mathematical functions
import matplotlib.pyplot as plt # Used for creating static, animated, and interactive visualizations in Python

from sklearn.model_selection import train_test_split, cross_val_score
# train_test_split: Used for splitting the dataset into training and testing sets
# cross_val_score: Used for evaluating a model's performance using cross-validation

from sklearn.neighbors import KNeighborsClassifier # Used for implementing the k-nearest neighbors (KNN) classification algorithm
from sklearn.preprocessing import StandardScaler # Used for standardizing features by removing the mean and scaling to unit variance
from sklearn.metrics import accuracy_score # Used for calculating the accuracy of the model's predictions

# Load and prepare the data
data = pd.read_csv('fruits.csv')
X = data[['color_score', 'width', 'height']] # Features
y = data['fruit_label'] # Target variable
fruit_names = data['fruit_name'].unique()

# Define color map for fruits
fruit_colors = {1: 'red', 2: 'yellow', 3: 'orange', 4: 'pink'}

# Split the dataset into training and testing sets
# X is the feature matrix containing the input data
# y is the target variable containing the class labels
# test_size=0.20 specifies that 20% of the data should be used for testing, and the remaining 80% for training
# random_state=42 ensures reproducibility by setting a seed for the random number generator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize a StandardScaler to standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
# The transformed data is converted back to a DataFrame with the original column names
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

# Transform the test data using the already fitted scaler
# The transformed data is converted back to a DataFrame with the original column names
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

    # Determine the minimum and maximum values for feature1, with a margin of 1 unit for better visualization
    x_min, x_max = X[feature1].min() - 1, X[feature1].max() + 1
    
    # Determine the minimum and maximum values for feature2, with a margin of 1 unit for better visualization
    y_min, y_max = X[feature2].min() - 1, X[feature2].max() + 1
    
    # Create a mesh grid for plotting the decision boundary
    # np.arange generates values from x_min to x_max with a step of 0.1 for the x-axis
    # np.arange generates values from y_min to y_max with a step of 0.1 for the y-axis
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    
    # Initialize the KNeighborsClassifier with the specified number of neighbors (k) and the Euclidean distance metric
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    
    # Fit the KNN classifier on the training data
    # X[[feature1, feature2]] selects the columns corresponding to feature1 and feature2 from the dataset X
    # y is the target variable containing the class labels
    knn.fit(X[[feature1, feature2]], y)
    

    # Create a DataFrame from the mesh grid for prediction
    # The DataFrame has two columns: feature1 and feature2, created from the raveled (flattened) mesh grid arrays xx and yy
    mesh_df = pd.DataFrame({feature1: xx.ravel(), feature2: yy.ravel()})
    
    # Predict the class labels for each point in the mesh grid using the trained KNN classifier
    # The predictions are reshaped to match the shape of the mesh grid for plotting
    Z = knn.predict(mesh_df).reshape(xx.shape)
    
    # Plot decision boundaries using a filled contour plot
    # xx and yy are the mesh grid arrays, and Z contains the predicted class labels for each point in the mesh grid
    # alpha=0.4 sets the transparency level of the contour plot
    # cmap=plt.cm.RdYlBu specifies the colormap to use for the plot
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Scatter plot of the original data points
    # X[feature1] and X[feature2] are the feature columns used for plotting
    # c=[fruit_colors[label] for label in y] sets the color of each point based on its class label using the fruit_colors dictionary
    # alpha=0.8 sets the transparency level of the scatter plot points
    ax.scatter(X[feature1], X[feature2], c=[fruit_colors[label] for label in y], alpha=0.8)
    
    prediction = None
    
    if new_data is not None:
        # Scatter plot the new data point with a green star marker
        # s=200 sets the size of the marker
        ax.scatter(new_data[feature1], new_data[feature2], color='green', marker='*', s=200)
        
        # Find k nearest neighbors for the new data point
        # distances contains the distances to the k nearest neighbors
        # indices contains the indices of the k nearest neighbors in the training data
        distances, indices = knn.kneighbors(new_data[[feature1, feature2]])
        
        # Plot broken lines from the new data point to each of its k nearest neighbors
        for i in range(k):
            ax.plot([new_data[feature1].values[0], X[feature1].iloc[indices[0][i]]],
                    [new_data[feature2].values[0], X[feature2].iloc[indices[0][i]]],
                    'k--', alpha=0.3)
        
        # Print the average distance to the nearest neighbors (commented out)
        avg_distance = np.mean(distances)
        print(f'Average distance to the {k} nearest neighbors: {avg_distance}')
        
        # Predict the class label for the new data point
        prediction = knn.predict(new_data[[feature1, feature2]])[0]
    
    
    # Set the label for the x-axis of the plot to the name of feature1
    ax.set_xlabel(feature1)
    
    # Set the label for the y-axis of the plot to the name of feature2
    ax.set_ylabel(feature2)
    
    return prediction


# Calculate figure size to fit within 1080px vertical height
fig_width = 8  # Adjust this value to change the aspect ratio
fig_height = min(fig_width * 0.75, 10)  # Limit height to 1080px (assuming 108 DPI)

# Set up the plot with a 3x3 grid of subplots
# fig is the Figure object, and axs is an array of Axes objects
fig, axs = plt.subplots(3, 3, figsize=(fig_width, fig_height))

# Set the main title for the entire figure
# fontsize=14 sets the font size of the title
fig.suptitle('KNN Decision Boundaries and Accuracy Scores', fontsize=14)

# Define pairs of features to be used for plotting decision boundaries
# Each tuple contains two feature names
feature_pairs = [('width', 'height'), ('color_score', 'width'), ('color_score', 'height')]

# Define the values of k (number of neighbors) to be used for the KNN classifier
k_values = [1, 3, 5]

# Create plots and calculate accuracies for each combination of feature pairs and k values
for i, (f1, f2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        # Select the subplot for the current combination of feature pair and k value
        ax = axs[i, j]
        
        # Plot the KNN decision boundary for the current feature pair and k value
        plot_knn(ax, X_train_scaled, y_train, k, f1, f2)
        
        # Initialize the KNeighborsClassifier with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Calculate cross-validation accuracy using 5-fold cross-validation
        # cv_scores contains the accuracy scores for each fold
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)
        
        # Calculate the mean cross-validation accuracy
        mean_cv_acc = np.mean(cv_scores)
        
        # Fit the KNN classifier on the training data for the current feature pair
        knn.fit(X_train_scaled[[f1, f2]], y_train)
        
        # Predict the class labels for the test data
        y_pred = knn.predict(X_test_scaled[[f1, f2]])
        
        # Calculate the test accuracy
        test_acc = accuracy_score(y_test, y_pred)
        
        # Set the title of the subplot to display the k value, cross-validation accuracy, and test accuracy
        ax.set_title(f'K={k}\nCV Acc: {mean_cv_acc*100:.2f}%\nTest Acc: {test_acc*100:.2f}%', fontsize=9)

# Adjust the layout of the plots to prevent overlap and display them
plt.tight_layout()
plt.show()

# New data point to be classified
# The DataFrame contains one row with values for 'color_score', 'width', and 'height'
new_data = pd.DataFrame([[0.82, 7.7, 7.9]], columns=['color_score', 'width', 'height'])

# Scale the new data point using the previously fitted scaler
# The transformed data is converted back to a DataFrame with the original column names
new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)

# Set up the plot with a 3x3 grid of subplots for the new data point
# fig is the Figure object, and axs is an array of Axes objects
fig, axs = plt.subplots(3, 3, figsize=(fig_width, fig_height))

# Set the main title for the entire figure
# fontsize=14 sets the font size of the title
fig.suptitle('KNN Decision Boundaries with New Data Point', fontsize=14)

# Loop through each combination of feature pairs and k values to plot decision boundaries and classify the new data point
for i, (f1, f2) in enumerate(feature_pairs):
    for j, k in enumerate(k_values):
        # Select the subplot for the current combination of feature pair and k value
        ax = axs[i, j]
        
        # Plot the KNN decision boundary for the current feature pair and k value, including the new data point
        prediction = plot_knn(ax, X_train_scaled, y_train, k, f1, f2, new_data_scaled)
        
        # Determine the predicted fruit name based on the prediction
        # Assuming fruit labels start from 1, subtract 1 to get the correct index in the fruit_names list
        fruit_name = fruit_names[prediction - 1]
        
        # Set the title of the subplot to display the k value, predicted fruit name, and feature pair
        ax.set_title(f'K={k}: {fruit_name}\n{f1}, {f2}', fontsize=10)

# Adjust the layout of the plots to prevent overlap and display them
plt.tight_layout()
plt.show()


# Calculate and store accuracy scores for different feature pairs and k values
accuracy_data = []

# Loop through each combination of feature pairs
for f1, f2 in feature_pairs:
    # Loop through each k value
    for k in k_values:
        # Initialize the KNeighborsClassifier with the current k value
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Calculate cross-validation accuracy using 5-fold cross-validation
        # cv_scores contains the accuracy scores for each fold
        cv_scores = cross_val_score(knn, X_train_scaled[[f1, f2]], y_train, cv=5)
        
        # Calculate the mean cross-validation accuracy
        mean_cv_acc = np.mean(cv_scores)
        
        # Fit the KNN classifier on the training data for the current feature pair
        knn.fit(X_train_scaled[[f1, f2]], y_train)
        
        # Predict the class labels for the test data
        y_pred = knn.predict(X_test_scaled[[f1, f2]])
        
        # Calculate the test accuracy
        test_acc = accuracy_score(y_test, y_pred)
        
        # Store the accuracy data in a dictionary
        accuracy_data.append({
            'Features': f'{f1}, {f2}',  # The feature pair used
            'K': k,                      # The k value used
            'CV Accuracy': mean_cv_acc,  # The mean cross-validation accuracy
            'Test Accuracy': test_acc    # The test accuracy
        })

# The prediction is done again after fitting the model to the training data to calculate the test accuracy.
# Cross-validation accuracy provides an estimate of the model's performance on unseen data by splitting the training data into folds.
# The test accuracy is calculated on a separate test dataset to evaluate the model's performance on truly unseen data.
# This helps in understanding how well the model generalizes to new data.


# Create a DataFrame from the accuracy data
# accuracy_data is a list of dictionaries containing accuracy information for different feature pairs and k values
accuracy_df = pd.DataFrame(accuracy_data)

# Set up the figure for the bar plot with a specified size
plt.figure(figsize=(15, 10))

# Define the width of the bars in the bar plot
bar_width = 0.35

# Create an array of indices for the x-axis based on the number of rows in the accuracy DataFrame
index = np.arange(len(accuracy_df))

# Create a bar plot for cross-validation (CV) accuracies
# index specifies the positions of the bars on the x-axis
# accuracy_df['CV Accuracy'] provides the heights of the bars
# bar_width specifies the width of the bars
# label='CV Accuracy' sets the label for the legend
# alpha=0.8 sets the transparency level of the bars
plt.bar(index, accuracy_df['CV Accuracy'], bar_width, label='CV Accuracy', alpha=0.8)

# Create a bar plot for test accuracies
# index + bar_width shifts the positions of the bars to the right to avoid overlap with the CV accuracy bars
# accuracy_df['Test Accuracy'] provides the heights of the bars
# bar_width specifies the width of the bars
# label='Test Accuracy' sets the label for the legend
# alpha=0.8 sets the transparency level of the bars
plt.bar(index + bar_width, accuracy_df['Test Accuracy'], bar_width, label='Test Accuracy', alpha=0.8)

# Set the label for the x-axis of the plot
plt.xlabel('Feature Pairs and K Values')

# Set the label for the y-axis of the plot
plt.ylabel('Accuracy')

# Set the title of the plot
plt.title('Comparison of Cross-Validation and Test Accuracies')

# Set the tick labels for the x-axis
# index + bar_width / 2 centers the tick labels between the pairs of bars
# [f"{row['Features']}, K={row['K']}" for _, row in accuracy_df.iterrows()] generates the tick labels from the feature pairs and k values
# rotation=45 rotates the tick labels by 45 degrees for better readability
# ha='right' aligns the tick labels to the right
plt.xticks(index + bar_width / 2, [f"{row['Features']}, K={row['K']}" for _, row in accuracy_df.iterrows()], rotation=45, ha='right')

# Display the legend for the plot
plt.legend()

# Adjust the layout of the plot to prevent overlap and display it
plt.tight_layout()
plt.show()

# Print accuracy scores
print("\nAccuracy Scores:")
for _, row in accuracy_df.iterrows():
    print(f"\nFeatures: {row['Features']}, K={row['K']}")
    print(f"CV Accuracy = {row['CV Accuracy']*100:.2f}%, Test Accuracy = {row['Test Accuracy']*100:.2f}%")