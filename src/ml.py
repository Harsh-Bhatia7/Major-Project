import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
import seaborn as sns
import logging
import colorlog
import numpy as np
import joblib
import os

# Configure colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
for hdlr in logger.handlers[:]:
    if not isinstance(hdlr, colorlog.StreamHandler):
        logger.removeHandler(hdlr)

# Step 1: Define project and goals
logger.info("Step 1: Project definition - Iris flower classification using Decision Tree")

# Step 2: Data Collection
logger.info("Step 2: Data Collection - Loading Iris dataset")
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

logger.info(f"Dataset shape: {iris_df.shape}")
logger.info(f"First few rows:\n{iris_df.head()}")

# Step 3: Data Preprocessing
logger.info("Step 3: Data Preprocessing")
missing_values = iris_df.isnull().sum().sum()
logger.info(f"Total missing values: {missing_values}")

# Feature Scaling
logger.info("Performing feature scaling")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
logger.info("Feature scaling completed")

# Step 4: Feature Selection/Engineering
logger.info("Step 4: Feature Engineering - Applying PCA")
# Feature Selection - Using PCA for dimensionality reduction
X = iris.data
y = iris.target

pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)
pca_feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
logger.info(f"PCA components shape: {X_pca.shape}")
logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
logger.info(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Visualize PCA
logger.info("Visualizing PCA results")
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis', edgecolor='k')
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Species')
plt.tight_layout()
plt.savefig('pca_visualization.png')
plt.show()

# Step 5: Model Selection and Training
logger.info("Step 5: Model Selection - Decision Tree")

# Cross-validation to check for potential overfitting
logger.info("Performing cross-validation")
dt_cv = DecisionTreeClassifier(random_state=42)
cv_scores = cross_val_score(dt_cv, X_pca, y, cv=5)
logger.info(f"Cross-validation scores: {cv_scores}")
logger.info(f"Mean CV accuracy: {np.mean(cv_scores)*100:.2f}%")
logger.info(f"CV accuracy std dev: {np.std(cv_scores)*100:.2f}%")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")

# Initialize and train the Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=3)  # Added max_depth to prevent overfitting
logger.info("Training the model")
decision_tree.fit(X_train, y_train)
logger.info("Model training completed")

# Step 6: Model Evaluation
logger.info("Step 6: Testing and evaluating the model")
y_pred = decision_tree.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
logger.info(f"Model accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
logger.info(f"Confusion Matrix:\n{conf_matrix}")

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
logger.info(f"Classification Report:\n{class_report}")

# Visualize the decision tree with correct PCA feature names
logger.info("Visualizing the decision tree with PCA features")
plt.figure(figsize=(12, 8))
tree.plot_tree(decision_tree, feature_names=pca_feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization (PCA Features)")
plt.tight_layout()
plt.savefig('decision_tree.png')
plt.show()

# Step 7: Save the model and preprocessors
logger.info("Step 7: Saving the trained model and preprocessors")
model_dir = os.path.join(os.getcwd(), 'iris_model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the model, scaler, and PCA
joblib.dump(decision_tree, os.path.join(model_dir, 'decision_tree_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(pca, os.path.join(model_dir, 'pca.pkl'))
logger.info(f"Model and preprocessors saved to {model_dir} directory")

# Step 8: Model Implementation - Function to use the trained model for predictions
logger.info("Step 8: Implementing the model for predictions")

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict the iris species based on the flower's dimensions.
    
    Parameters:
    -----------
    sepal_length : float
        Length of the sepal in cm
    sepal_width : float
        Width of the sepal in cm
    petal_length : float
        Length of the petal in cm
    petal_width : float
        Width of the petal in cm
        
    Returns:
    --------
    species_name : str
        Predicted species name
    """
    # Load the saved model and preprocessors
    model = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
    saved_scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    saved_pca = joblib.load(os.path.join(model_dir, 'pca.pkl'))
    
    # Create a feature array from input
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the features
    features_scaled = saved_scaler.transform(features)
    
    # Apply PCA
    features_pca = saved_pca.transform(features_scaled)
    
    # Make prediction
    species_idx = model.predict(features_pca)[0]
    species_name = iris.target_names[species_idx]
    
    return species_name

# Example usage
logger.info("Example prediction with the saved model:")
# Sample from the test data
sample_idx = 0
sample = X_test[sample_idx]
true_species = iris.target_names[y_test[sample_idx]]

# Get original features for this sample
original_idx = np.where((iris.target == y_test[sample_idx]))[0][0]
original_features = iris.data[original_idx]

sepal_length, sepal_width, petal_length, petal_width = original_features
predicted_species = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)

logger.info(f"Sample features: Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, "
            f"Petal Length: {petal_length}, Petal Width: {petal_width}")
logger.info(f"True species: {true_species}")
logger.info(f"Predicted species: {predicted_species}")

logger.info("Iris Decision Tree Classification workflow completed successfully")