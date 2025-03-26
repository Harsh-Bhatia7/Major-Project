import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureProcessor:
    """Class to handle feature processing operations"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger
        self.scaler = None
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.pca_feature_names = None

    def scale_features(self, X):
        """Scale features using StandardScaler"""
        self.logger.info("Performing feature scaling")
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.logger.info("Feature scaling completed")
        return self.X_scaled

    def apply_pca(self, X_scaled, n_components=2):
        """Apply PCA for dimensionality reduction"""
        self.logger.info(f"Step 4: Feature Engineering - Applying PCA with {n_components} components")
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_scaled)
        self.pca_feature_names = [f'PC{i + 1}' for i in range(self.X_pca.shape[1])]

        self.logger.info(f"PCA components shape: {self.X_pca.shape}")
        self.logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        self.logger.info(f"Total variance explained: {sum(self.pca.explained_variance_ratio_) * 100:.2f}%")
        return self.X_pca, self.pca_feature_names

    def visualize_pca(self, X_pca, y, target_names=None):
        """Visualize PCA results"""
        self.logger.info("Visualizing PCA results")
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
        plt.title('PCA of Iris Dataset')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Species')
        plt.tight_layout()
        plt.savefig('pca_visualization.png')
        plt.show()
