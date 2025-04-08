from setuptools import setup, find_packages

setup(
    name="iris_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "numpy",
        "colorlog",
        "joblib",
        "streamlit",
    ],
    author="Harsh Bhatia",
    author_email="harshbhatia0007@gmail.com",
    description="Iris flower classification using machine learning",
    keywords="machine learning, classification, iris dataset",
    python_requires=">=3.6",
)
