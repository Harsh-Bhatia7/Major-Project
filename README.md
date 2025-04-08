# Iris Classifier Package

A comprehensive machine learning package for classifying iris flowers by species using the classic Iris dataset.

## Overview

This package provides a complete workflow for iris flower classification using multiple ML models (Decision Tree, Random Forest, SVM, and KNN). It includes data loading from multiple sources, feature processing with PCA, model training with cross-validation, evaluation, visualization, and an interactive web interface.

## Features

- **Data Loading**: Load and preprocess the Iris dataset from CSV or SQLite database
- **Feature Processing**: Scale features and apply PCA dimensionality reduction
- **Model Selection**: Choose from Decision Tree, Random Forest, SVM, or KNN classifiers
- **Model Training**: Train models with cross-validation and hyperparameter tuning
- **Model Comparison**: Compare performance of different models and select the best one
- **Model Evaluation**: Evaluate model performance with accuracy, confusion matrix, and classification report
- **Visualization**: Generate PCA plots, confusion matrices, and decision tree visualizations
- **Model Storage**: Save and load trained models and preprocessors
- **Prediction**: Predict iris species based on flower measurements
- **Interactive Web Interface**: User-friendly Streamlit interface for making predictions
- **Comprehensive Testing**: Extensive unit tests for all components and pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/KrishTalwar03/Major-Project.git
cd Major-Project

# Install the package and dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
python -m src.main
```

### Web Interface

```bash
# Launch the Streamlit web interface
streamlit run frontend/gui.py
```

### Making Predictions Programmatically

```python
from src import IrisClassifier

# Initialize the classifier with your preferred model
classifier = IrisClassifier(model_type="random_forest")

# Train the model (or load a saved model)
classifier.run_full_workflow()

# Predict the species for a new flower
species = classifier.predict_iris_species(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)
print(f"Predicted species: {species}")
```

### Running Tests

The package includes comprehensive unit tests to ensure all components work correctly.

```bash
# Run all unit tests with the custom test runner
python -m unit_tests.run_all_tests
```

## Available Models

- `'decision_tree'`: Decision Tree Classifier
- `'random_forest'`: Random Forest Classifier
- `'svm'`: Support Vector Machine
- `'knn'`: K-Nearest Neighbors

## Project Structure

```
Major-Project/
├── dataset/
│   └── iris_species/
│       ├── Iris.csv          # Original CSV dataset
│       └── database.sqlite   # SQLite database
├── frontend/
│   └── gui.py                # Streamlit web interface
├── src/
│   ├── __init__.py
│   ├── classifier.py         # Main classifier class
│   ├── data_loading.py       # Data loading utilities
│   ├── feature_proc.py       # Feature processing
│   ├── logger.py             # Colored logging setup
│   ├── model_eval.py         # Model evaluation tools
│   ├── model_storage.py      # Model saving/loading
│   ├── model_train.py        # Model training utilities
│   └── main.py               # Entry point
├── unit_tests/               # Unit test package
│   ├── __init__.py
│   ├── custom_test_runner.py # Custom test runner with improved output
│   ├── run_all_tests.py      # Test runner script
│   ├── test_classifier_functions.py  # Tests for classifier components
│   ├── test_database.py      # Tests for database functionality
│   ├── test_performance.py   # Performance benchmark tests
│   ├── test_pipeline.py      # Pipeline integration tests
│   └── test_ui_navigation.py # UI navigation tests
├── README.md
└── setup.py
```

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- colorlog
- joblib
- streamlit
- psutil (for performance tests)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

@Harsh-Bhatia7 - harshbhatia0007@gmail.com
@KrishTalwar03 - krishtalwar271@gmail.com