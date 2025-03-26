from .classifier import IrisClassifier


def main():
    """Main entry point for the application"""
    classifier = IrisClassifier()
    classifier.run_full_workflow()


if __name__ == "__main__":
    main()
