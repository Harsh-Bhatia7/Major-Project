import unittest
import os
import sys
import pandas as pd
import sqlite3
import tempfile
import shutil
import matplotlib
from src.data_loading import DataLoader
from src.logger import Logger
matplotlib.use('Agg')  # Disable plots from showing during tests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDatabase(unittest.TestCase):
    """Test database functionality of the Iris Classifier."""

    def setUp(self):
        """Set up test environment."""
        self.logger = Logger()
        self.data_loader = DataLoader(self.logger)

        # Create a temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, "test_database.sqlite")

        # Create a sample dataframe
        self.sample_df = pd.DataFrame({
            'Id': [1, 2, 3],
            'SepalLengthCm': [5.1, 4.9, 4.7],
            'SepalWidthCm': [3.5, 3.0, 3.2],
            'PetalLengthCm': [1.4, 1.4, 1.3],
            'PetalWidthCm': [0.2, 0.2, 0.2],
            'Species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa']
        })

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)


def test_create_database_from_csv(self):
    """Test creating a database from CSV data."""
    # Override csv_path with our dataframe
    self.data_loader.csv_path = None

    # Create a temporary CSV file
    csv_path = os.path.join(self.temp_dir, "test.csv")
    self.sample_df.to_csv(csv_path, index=False)

    # Create database from the CSV
    conn = sqlite3.connect(self.temp_db)

    try:
        # Write dataframe to database
        self.sample_df.to_sql('iris', conn, if_exists='replace', index=False)

        # Create an index explicitly
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_species ON iris (Species)")
        conn.commit()

        # Verify data was written correctly
        query = "SELECT * FROM iris"
        df_from_db = pd.read_sql_query(query, conn)

        # Check if dataframes are equal
        pd.testing.assert_frame_equal(df_from_db, self.sample_df)

        # Check if index was created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = cursor.fetchall()

        # There should be at least one index
        self.assertTrue(len(indexes) >= 1)

    finally:
        conn.close()

    def test_load_from_database(self):
        """Test loading data from the database."""
        # Create a database with test data
        conn = sqlite3.connect(self.temp_db)
        self.sample_df.to_sql('iris', conn, if_exists='replace', index=False)
        conn.close()

        # Override db_path with our test database
        self.data_loader.db_path = self.temp_db

        # Load data from the database
        iris, iris_df = self.data_loader._load_from_database(self.temp_db)

        # Check that data was loaded correctly
        self.assertEqual(len(iris_df), 3)
        self.assertEqual(list(iris_df['Species']), ['Iris-setosa', 'Iris-setosa', 'Iris-setosa'])

        # Check that the iris object was created correctly
        self.assertEqual(iris.data.shape, (3, 4))
        self.assertEqual(len(iris.target), 3)
        self.assertEqual(list(iris.target), [0, 0, 0])  # All setosa (0)

    def test_database_fallback_to_csv(self):
        """Test that loading falls back to CSV if database fails."""
        # Set the db_path to a non-existent file
        non_existent_db = os.path.join(self.temp_dir, "nonexistent.sqlite")

        # Create a temporary CSV file
        csv_path = os.path.join(self.temp_dir, "test.csv")
        self.sample_df.to_csv(csv_path, index=False)

        # Set paths
        self.data_loader.db_path = non_existent_db
        self.data_loader.csv_path = csv_path

        # This should create the database from CSV since it doesn't exist
        iris, iris_df = self.data_loader.load_iris_dataset(source_type="db")

        # Verify data was loaded correctly
        self.assertEqual(len(iris_df), 3)

        # The database should now exist and have data
        self.assertTrue(os.path.exists(non_existent_db))


if __name__ == '__main__':
    unittest.main()
