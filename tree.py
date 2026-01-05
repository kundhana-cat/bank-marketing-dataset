# Task 03: Decision Tree Classifier 

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("bank-full.csv", sep=";")

print("First 5 rows of dataset:")
print(data.head())

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop("y", axis=1)
y = data["y"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

# Create Decision Tree model using entropy
dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=6,
    random_state=1
)

# Train model
dt_model.fit(X_train, y_train)

# Predict
y_pred = dt_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize Decision Tree
plt.figure(figsize=(18, 8))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree Classifier (Entropy)")
plt.show()
