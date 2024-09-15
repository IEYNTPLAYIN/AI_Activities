import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
data = pd.read_csv('edited-fruits.csv')

X = data[['mass', 'width', 'height', 'color_score']]  # Features
y = data['fruit_label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal k using cross-validation
k_values = (1, 3, 5) 
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f'Cross-validation accuracy for k={k}: {cv_scores[-1]:.4f}')


optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f'Optimal K: {optimal_k}')

# Train final model with optimal k
final_model = KNeighborsClassifier(n_neighbors=optimal_k)
final_model.fit(X_train_scaled, y_train)

# Evaluate on test set
test_predictions = final_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test set accuracy with k={optimal_k}: {test_accuracy:.4f}%')


# Visualizations

# 1. K selection plot
plt.figure(figsize=(8, 4))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Value of K')
plt.ylabel('Cross-validation Accuracy')
plt.title('K Selection based on Cross-validation')
plt.show()

# 2. Pair plot for all features
sns.pairplot(data, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])
plt.show()

# # 3. Confusion Matrix
# cm = confusion_matrix(y_test, test_predictions)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix (Test Set)')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # 4. Feature Importance (based on correlation with target)
# corr = data.corr()['fruit_label'].drop('fruit_label')
# plt.figure(figsize=(8, 4))
# corr.plot(kind='bar')
# plt.title('Feature Importance (Correlation with Target)')
# plt.show()