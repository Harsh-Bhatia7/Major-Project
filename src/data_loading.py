import pandas as pd
import os
import sqlite3


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
        self.csv_path = "dataset\\iris_species\\Iris.csv"
        self.db_path = "dataset\\iris_species\\database.sqlite"

    def load_iris_dataset(self, source_type="db", db_path=None, csv_path=None):
        """
        Load the iris dataset from SQLite database or CSV file

        Args:
            source_type (str): 'db' for database or 'csv' for CSV file
            db_path (str, optional): Path to the SQLite database
            csv_path (str, optional): Path to the iris CSV file
        """
        self.logger.info("Step 2: Data Collection - Loading Iris dataset")

        if source_type == "db":
            return self._load_from_database(db_path)
        else:
            return self._load_from_csv(csv_path)

    def _load_from_database(self, db_path=None):
        """Load the Iris dataset from SQLite database"""
        # Define the default path if not provided
        if db_path is None:
            db_path = self.db_path

        try:
            # Check if database exists, if not, create and populate it
            if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
                self.logger.info(f"Database not found or empty. Creating and populating from CSV: {db_path}")
                self._create_database_from_csv(db_path)

            # Connect to the SQLite database
            self.logger.info(f"Loading data from database: {db_path}")
            conn = sqlite3.connect(db_path)

            # Load the data from the database
            query = "SELECT * FROM iris"
            self.iris_df = pd.read_sql_query(query, conn)
            conn.close()

            # Process the loaded dataframe
            return self._process_dataframe()

        except Exception as e:
            self.logger.error(f"Error loading from database: {str(e)}. Falling back to CSV.")
            return self._load_from_csv()

    def _create_database_from_csv(self, db_path):
        """Create a new SQLite database and populate it with data from the CSV file"""
        try:
            # Load CSV data
            csv_df = pd.read_csv(self.csv_path)

            # Create SQLite database and connection
            conn = sqlite3.connect(db_path)

            # Write the dataframe to a SQLite table
            csv_df.to_sql('iris', conn, if_exists='replace', index=False)

            # Create indexes for faster querying
            cursor = conn.cursor()
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON iris(Species)')
            conn.commit()
            conn.close()

            self.logger.info(f"Successfully created database from CSV at {db_path}")
        except Exception as e:
            self.logger.error(f"Error creating database from CSV: {str(e)}")
            raise

    def _load_from_csv(self, csv_path=None):
        """Load the Iris dataset from CSV file"""
        # Define the default path if not provided
        if csv_path is None:
            csv_path = self.csv_path

        try:
            # Load the data from CSV
            self.logger.info(f"Loading data from CSV: {csv_path}")
            self.iris_df = pd.read_csv(csv_path)

            # Process the loaded dataframe
            return self._process_dataframe()

        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}. Unable to load dataset.")
            raise

    def _process_dataframe(self):
        """Process the loaded dataframe and set up the iris object"""
        # Map species to numeric target values (0, 1, 2)
        species_mapping = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 2
        }

        # Define feature column names from the dataframe
        self.feature_names = [
            'SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm'
        ]

        # Get target names
        species_col = 'Species'
        self.target_names = list(self.iris_df[species_col].unique())

        # Map species strings to integers
        self.iris_df['target'] = self.iris_df[species_col].map(species_mapping)

        # Extract features and target
        self.X = self.iris_df[self.feature_names].values
        self.y = self.iris_df['target'].values

        # Create a structure similar to what the rest of the code expects
        class IrisDataset:
            pass

        self.iris = IrisDataset()
        self.iris.data = self.X
        self.iris.target = self.y
        self.iris.feature_names = self.feature_names
        self.iris.target_names = self.target_names

        # Optional: Create more descriptive feature names like sklearn's
        self.iris.feature_names_descriptive = [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ]

        self.logger.info("Successfully loaded iris dataset")
        self.logger.info(f"Dataset shape: {self.iris_df.shape}")
        self.logger.info(f"First few rows:\n{self.iris_df.head()}")
        return self.iris, self.iris_df

    def check_missing_values(self):
        """Check for missing values in the dataset"""
        missing_values = self.iris_df.isnull().sum().sum()
        self.logger.info(f"Total missing values: {missing_values}")
        return missing_values
