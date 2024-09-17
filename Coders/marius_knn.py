import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('edited-fruits.csv')

X = data[['mass', 'color_score']]
y = data['fruit_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

knn = KNeighborsClassifier(n_neighbors=10, weights='distance')

knn.fit(X_train_scaled, y_train)

X_test_df = pd.DataFrame(X_test_scaled, columns=['mass', 'color_score'])

y_pred = knn.predict(X_test_df)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# New data
new_data = np.array([[180, 0.69]])
new_data_scaled = scaler.transform(new_data)

distances, indices = knn.kneighbors(new_data_scaled)

prediction = knn.predict(new_data_scaled)
print(f"Predicted class for the new data: {prediction[0]}")



# Plotting
plt.figure(figsize=(10, 6))
plt.style.use('bmh')

fruit_colors = {'apple':'red', 'orange':'orange', 'mandarin':'peru', 'lemon':'yellow'}
for fruit, color in fruit_colors.items():
    mask = y_train == fruit
    plt.scatter(X_train['mass'][mask], X_train['color_score'][mask], color=color, label=fruit.capitalize(), s=100, edgecolor='none')

# Plot the new data point
plt.scatter(new_data[0, 0], new_data[0, 1], color='blue', marker='*', s=200, label='New data')

for i, index in enumerate(indices[0]):
    neighbor = X_train.iloc[index]
    distance = distances[0][i]
    line_width = 1  # Uniform line width for neighbor lines
    plt.scatter(neighbor['mass'], neighbor['color_score'], facecolors=fruit_colors[y_train.iloc[index]],
                edgecolors='black', s=200, linewidths=2, label=f'Neighbor {i+1}' if i == 0 else None)
    plt.plot([new_data[0, 0], neighbor['mass']], [new_data[0, 1], neighbor['color_score']], 'k--', linewidth=line_width)

plt.title(f"KNN classification with {len(indices[0])} neighbors (Distance-weighted)")
plt.xlabel('Mass')
plt.ylabel('Color Score')

plt.legend(title="Fruit Types", loc='upper right', fontsize='10', frameon=True)

plt.grid(True)
plt.show()
