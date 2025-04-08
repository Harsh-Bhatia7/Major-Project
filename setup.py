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
        "psutil",  # Required for performance tests
    ],
    author="Harsh Bhatia, Krish Talwar",
    author_email="harshbhatia0007@gmail.com, krishtalwar271@gmail.com",
    description="Iris flower classification using machine learning",
    keywords="machine learning, classification, iris dataset",
    python_requires=">=3.6",
    test_suite="unit_tests",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
