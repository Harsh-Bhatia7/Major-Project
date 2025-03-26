# Step 1:- Select the dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
df = pd.read_csv("C:\\iris dataset\\iris.csv", encoding='utf-8')
print(df)

# Step 2:- Data Collection
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target column (species)
iris_df['species'] = iris.target

# Display the first few rows of the DataFrame
print(iris_df.head())

# Step 3:- Data Preprocessing

# Step 3:- part 1 to check the missing values
print(iris_df.isnull().sum())

# Step 3:- part 2 Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)

#Step 4: Feature Selection
from sklearn.decomposition import PCA

# 1) Independent Variables (Features)
X = iris.data  # Independent variables (sepal length, sepal width, petal length, petal width)

# 2) Dependent Variable (Target)
y = iris.target  # Dependent variable (species)

# Convert to DataFrame for easier understanding
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = y

print("Independent Variables (Features):")
print(iris_df.iloc[:, :-1].head())  # Display first few rows of the features (independent variables)

print("\nDependent Variable (Target):")
print(iris_df['species'].head())  # Display the target variable (species)
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Visualize PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.title('PCA of Iris Dataset')
plt.show()

# Step 5: Model Selection - Decision Tree

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca, iris.target, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Step 6: Train the model
decision_tree.fit(X_train, y_train)

# Step 7: Test the model and evaluate its performance
y_pred = decision_tree.predict(X_test)

# Evaluate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(12, 8))
tree.plot_tree(decision_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()