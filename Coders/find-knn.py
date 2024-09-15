import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
data = pd.read_csv('./Assets/fruits.csv')

X = data[['mass', 'width', 'height', 'color_score']]  # features
y = data['fruit_label']  # target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal k using cross-validation
k_values = (1, 3, 5)
accuracies = []

# make predictions for each value of k

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=3, scoring='accuracy')
    accuracies.append(scores.mean())
    print(f'Accuracy for k={k}: {accuracies[-1]:.4f}')

optimal_k = k_values[accuracies.index(max(accuracies))]

# Visualizations

# plt.figure(figsize=(10, 6))
# sns.barplot(x=k_values, y=accuracies)
# plt.xlabel('Value of K')
# plt.ylabel('Accuracy')
# plt.title('Comparison of K values')
# plt.show()

# # 1. Scatter plot for mass vs width
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='mass', y='width', hue='fruit_name')
# plt.title('Mass vs Width')
# plt.show()

# 2. Pair plot for all features
sns.pairplot(data, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])
plt.show()

# # 3. Histogram for mass
# plt.figure(figsize=(10, 6))
# sns.histplot(data=data, x='mass', hue='fruit_name', kde=True, multiple="stack")
# plt.title('Distribution of Mass by Fruit Type')
# plt.show()

# # 4. Confusion Matrix
# cm = confusion_matrix(y_test, predictions)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # 5. Feature Importance (based on correlation with target)
# corr = data.corr()['fruit_label'].drop('fruit_label')
# plt.figure(figsize=(10, 6))
# corr.plot(kind='bar')
# plt.title('Feature Importance (Correlation with Target)')
# plt.show()

# # 6. Elbow Curve for K selection
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracies)
# plt.xlabel('Value of K')
# plt.ylabel('Cross-validated Accuracy')
# plt.title('Elbow Curve for K Selection')
# plt.show()