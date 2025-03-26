import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class ModelEvaluator:
    """Class to handle model evaluation and visualization"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger

    def evaluate_model(self, model, X_test, y_test, target_names=None):
        """Evaluate model performance"""
        self.logger.info("Step 6: Testing and evaluating the model")
        y_pred = model.predict(X_test)

        # Evaluate the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Model accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Classification Report
        class_report = classification_report(y_test, y_pred, target_names=target_names)
        self.logger.info(f"Classification Report:\n{class_report}")

        return accuracy, conf_matrix, class_report, y_pred

    def plot_confusion_matrix(self, y_test, y_pred, target_names):
        """Visualize confusion matrix"""
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Species')
        plt.xlabel('Predicted Species')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

    def visualize_decision_tree(self, model, feature_names, class_names):
        """Visualize the decision tree"""
        self.logger.info("Visualizing the decision tree")
        plt.figure(figsize=(12, 8))
        tree.plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
        plt.title("Decision Tree Visualization")
        plt.tight_layout()
        plt.savefig('decision_tree.png')
        plt.show()
