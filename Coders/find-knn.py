import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Load and prepare the data
data = pd.read_csv('edited-fruits.csv')

X = data[['mass', 'width', 'height']]  # Features
y = data['fruit_label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# The purpose of setting random_state=42 in KNN is to ensure reproducibility of results


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform cross-validation for k=1, k=3, and k=5
k_values = [1, 3, 5]
cv_scores = []



for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=4)
    knn.fit(X_train, y_train)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f'Cross-validation accuracy for k={k}: {cv_scores[-1]:.4f}')
    
    
# Predict the test set results
y_pred = knn.predict(X_test)
print(y_pred)

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
print(cm)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'F1 Score: {f1:.4f}')
print(f'Accuracy score: {accuracy_score(y_test, y_pred):.4f}')



# Visualization of cross-validation results
plt.figure(figsize=(8, 4))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('Value of K')
plt.ylabel('Cross-validation Accuracy')
plt.title('KNN Performance for Different K Values')
plt.show()

# Additional visualization suggestions:

# 1. Pairplot of features
sns.pairplot(data, hue='fruit_name', vars=['mass', 'width', 'height'])
plt.suptitle('Pairplot of Fruit Features', y=1.02)
plt.show()

# 2. Box plot of mass by fruit type
plt.figure(figsize=(8, 4))
sns.boxplot(x='fruit_name', y='mass', data=data)
plt.title('Distribution of Mass by Fruit Type')
plt.show()

# 3. Scatter plot of width vs height, colored by fruit type
plt.figure(figsize=(8, 4))
sns.scatterplot(x='width', y='height', hue='fruit_name', data=data)
plt.title('Width vs Height by Fruit Type')
plt.show()

# # 4. Heatmap of feature correlations
# plt.figure(figsize=(8, 6))
# sns.heatmap(data[['mass', 'width', 'height']].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap of Features')
# plt.show()