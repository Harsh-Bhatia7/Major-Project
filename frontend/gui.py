import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.classifier import IrisClassifier


def main():
    st.set_page_config(page_title="Iris Classifier", layout="wide")
    st.title("🌸 Iris Flower Classification")

    # Initialize classifier
    @st.cache_resource
    def get_classifier():
        classifier = IrisClassifier()
        # Load saved model instead of running full workflow
        classifier.model_saver.create_model_directory()
        classifier.model, classifier.scaler, classifier.pca = classifier.model_saver.load_model_and_preprocessors()

        # Load iris dataset for display and reference
        classifier.iris, iris_df = classifier.data_loader.load_iris_dataset()
        return classifier, iris_df

    classifier, iris_df = get_classifier()

    # Sidebar with options
    st.sidebar.header("Options")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Dataset", "Model Info"])

    if page == "Prediction":
        st.header("Make a Prediction")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Enter Flower Measurements")
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

            if st.button("Predict Species"):
                species = classifier.predict_iris_species(
                    sepal_length, sepal_width, petal_length, petal_width
                )

                st.success(f"Predicted Species: **{species}**")

                # Show image of predicted species
                species_images = {
                    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
                    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
                }

                st.image(species_images[species], width=300)

        with col2:
            st.subheader("Feature Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot where the prediction falls in the feature space
            X_pca = classifier.pca.transform(classifier.scaler.transform(iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values))

            # Plot the PCA results
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=iris_df['Species'], palette='viridis', alpha=0.7, ax=ax)

            # Plot the new point
            new_features = classifier.pca.transform(classifier.scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]]))
            ax.scatter(new_features[:, 0], new_features[:, 1], color='red', s=100, marker='X', edgecolor='black', label='Your Input')

            ax.set_title('PCA of Iris Dataset')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.legend()
            st.pyplot(fig)

    elif page == "Dataset":
        st.header("Iris Dataset")
        st.dataframe(iris_df.head(50))

        st.subheader("Dataset Visualization")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Sepal length vs width
        sns.scatterplot(ax=axes[0, 0], x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris_df)
        axes[0, 0].set_title('Sepal Length vs Width')

        # Petal length vs width
        sns.scatterplot(ax=axes[0, 1], x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=iris_df)
        axes[0, 1].set_title('Petal Length vs Width')

        # Distribution of sepal length
        sns.boxplot(ax=axes[1, 0], x='Species', y='SepalLengthCm', data=iris_df)
        axes[1, 0].set_title('Sepal Length Distribution')

        # Distribution of petal length
        sns.boxplot(ax=axes[1, 1], x='Species', y='PetalLengthCm', data=iris_df)
        axes[1, 1].set_title('Petal Length Distribution')

        plt.tight_layout()
        st.pyplot(fig)

    else:  # Model Info
        st.header("Model Information")

        st.subheader("Model Performance")

        # Compare model performance
        results, best_model_name, _ = classifier.model_trainer.compare_models(
            classifier.feature_processor.X_pca,
            classifier.feature_processor.X_pca,
            classifier.iris.target,
            classifier.iris.target
        )

        # Display model comparison
        comparison_data = {
            'Model': [],
            'Training Accuracy (%)': [],
            'Testing Accuracy (%)': []
        }

        for name, result in results.items():
            comparison_data['Model'].append(name)
            comparison_data['Training Accuracy (%)'].append(result['train_accuracy'])
            comparison_data['Testing Accuracy (%)'].append(result['test_accuracy'])

        comparison_df = pd.DataFrame(comparison_data)
        st.write(comparison_df)

        st.subheader("Current Model: " + classifier.model_type)
        st.write(f"Best performing model: **{best_model_name}**")


if __name__ == "__main__":
    main()
