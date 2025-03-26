import pandas as pd
from sklearn.datasets import load_iris


class DataLoader:
    """Class to handle data loading and preprocessing"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger
        self.iris = None
        self.iris_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None

    def load_iris_dataset(self):
        """Load the iris dataset"""
        self.logger.info("Step 2: Data Collection - Loading Iris dataset")
        self.iris = load_iris()
        self.iris_df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.iris_df['species'] = self.iris.target

        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names

        self.logger.info(f"Dataset shape: {self.iris_df.shape}")
        self.logger.info(f"First few rows:\n{self.iris_df.head()}")
        return self.iris, self.iris_df

    def check_missing_values(self):
        """Check for missing values in the dataset"""
        missing_values = self.iris_df.isnull().sum().sum()
        self.logger.info(f"Total missing values: {missing_values}")
        return missing_values
