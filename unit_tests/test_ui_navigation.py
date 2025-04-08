import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import matplotlib
from frontend.gui import main
matplotlib.use('Agg')  # Disable plots from showing during tests

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestUINavigation(unittest.TestCase):
    """Tests for the UI navigation of the Streamlit application."""

    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.sidebar.header')
    @patch('streamlit.sidebar.selectbox')
    def test_app_initialization(self, mock_selectbox, mock_header, mock_title, mock_page_config):
        """Test that the app initializes correctly with proper sidebar navigation."""
        # Arrange
        mock_selectbox.return_value = "Prediction"

        # Instead of patching get_classifier directly, let's patch the IrisClassifier
        # Since main() will instantiate this class internally
        with patch('src.classifier.IrisClassifier') as mock_classifier_class:
            # Configure the mock classifier instance
            mock_classifier = MagicMock()
            mock_classifier.model_type = "decision_tree"
            mock_classifier.data_loader.load_iris_dataset.return_value = (MagicMock(), pd.DataFrame())
            mock_classifier_class.return_value = mock_classifier

            # Call the main function with mocked components
            main()

        # Assert
        mock_page_config.assert_called_once()
        mock_title.assert_called_once_with("🌸 Iris Flower Classification")
        mock_header.assert_called_once_with("Options")
        mock_selectbox.assert_called_once()
        self.assertEqual(mock_selectbox.call_args[0][0], "Choose a page")
        self.assertEqual(mock_selectbox.call_args[0][1], ["Prediction", "Dataset", "Model Info"])

    def test_prediction_page_loads(self):
        """Test that the prediction page loads correctly."""
        # Set up mock for IrisClassifier
        with patch('src.classifier.IrisClassifier') as mock_classifier_class:
            # Configure the mock classifier instance
            mock_classifier = MagicMock()
            mock_classifier.model_type = "decision_tree"
            mock_classifier.data_loader.load_iris_dataset.return_value = (MagicMock(), pd.DataFrame())
            mock_classifier_class.return_value = mock_classifier

            # Mock streamlit components
            with patch('streamlit.header') as mock_header:
                with patch('streamlit.sidebar.selectbox', return_value="Prediction"):
                    # Call main with our mocks
                    main()
                    mock_header.assert_any_call("Make a Prediction")

    @patch('streamlit.dataframe')
    def test_dataset_page_loads(self, mock_dataframe):
        """Test that the dataset page loads correctly."""
        # Set up mock for IrisClassifier
        with patch('src.classifier.IrisClassifier') as mock_classifier_class:
            # Configure the mock classifier
            mock_classifier = MagicMock()
            mock_iris_df = pd.DataFrame()
            mock_classifier.data_loader.load_iris_dataset.return_value = (MagicMock(), mock_iris_df)
            mock_classifier_class.return_value = mock_classifier

            # Mock streamlit components
            with patch('streamlit.sidebar.selectbox', return_value="Dataset"):
                # Run the test
                main()

        # Assert
        mock_dataframe.assert_called_once()

    @patch('streamlit.write')
    def test_model_info_page_loads(self, mock_write):
        """Test that the model info page loads correctly."""
        # Set up mock for IrisClassifier
        with patch('src.classifier.IrisClassifier') as mock_classifier_class:
            # Configure the mock classifier
            mock_classifier = MagicMock()
            mock_classifier.feature_processor = MagicMock()
            mock_classifier.feature_processor.X_pca = None
            mock_classifier.model_trainer = MagicMock()
            mock_classifier.model_trainer.split_data.return_value = (None, None, None, None)
            mock_classifier.model_trainer.compare_models.return_value = ({}, "decision_tree", None)
            mock_classifier.model_type = "decision_tree"
            mock_classifier.data_loader.load_iris_dataset.return_value = (MagicMock(), pd.DataFrame())
            mock_classifier_class.return_value = mock_classifier

            # Mock streamlit components
            with patch('streamlit.sidebar.selectbox', return_value="Model Info"):
                # Run the test
                main()

        # Assert
        mock_write.assert_called()


if __name__ == '__main__':
    unittest.main()
