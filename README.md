# Iris Classifier Package

A comprehensive machine learning package for classifying iris flowers by species using the classic Iris dataset.

## Overview

This package provides a complete workflow for iris flower classification using a Decision Tree model. It includes data loading, feature processing, model training, evaluation, and prediction capabilities.

## Features

- **Data Loading**: Load and preprocess the Iris dataset
- **Feature Processing**: Scale features and apply PCA dimensionality reduction
- **Model Training**: Train a Decision Tree classifier with cross-validation
- **Model Evaluation**: Evaluate model performance with accuracy, confusion matrix, and classification report
- **Visualization**: Generate PCA plots, confusion matrices, and decision tree visualizations
- **Model Storage**: Save and load trained models and preprocessors
- **Prediction**: Predict iris species based on flower measurements

## Installation

```bash
# Clone the repository
git clone https://github.com/KrishTalwar03/data-sci-Ces-2.git
cd data-sci-Ces-2

# Install the package and dependencies
pip install -e .
```

## Usage

### Basic Usage

```bash
python -m src.main
```

### Making Predictions

```python
# Predict the species for a new flower
species = classifier.predict_iris_species(
    sepal_length=5.1, 
    sepal_width=3.5, 
    petal_length=1.4, 
    petal_width=0.2
)
print(f"Predicted species: {species}")
```

## Project Structure

```
data-sci-Ces-2/
├── src/
│   ├── __init__.py
│   ├── classifier.py       # Main classifier class
│   ├── data_loading.py     # Data loading utilities
│   ├── feature_proc.py     # Feature processing
│   ├── logger.py           # Colored logging setup
│   ├── model_eval.py       # Model evaluation tools
│   ├── model_storage.py    # Model saving/loading
│   └── model_train.py      # Model training utilities
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

@Harsh-Bhatia7 - harshbhatia0007@gmail.com
@KrishTalwar03 - krishtalwar271@gmail.com