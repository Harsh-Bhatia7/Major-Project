import unittest
import sys
import os
import matplotlib
import logging
# Import test modules
from unit_tests.test_classifier_functions import TestClassifierFunctions
from unit_tests.test_performance import TestPerformance
from unit_tests.test_ui_navigation import TestUINavigation
from unit_tests.test_database import TestDatabase
from unit_tests.test_pipeline import TestPipeline
from unit_tests.custom_test_runner import MinimalTestRunner

# Disable plots from showing during tests
matplotlib.use('Agg')
# Disable debug logs
logging.disable(logging.CRITICAL)
matplotlib.use('Agg')  # Disable plots from showing during tests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_all_tests():
    """Run all test cases and generate a minimal report."""
    # Create a test suite containing all tests
    test_suite = unittest.TestSuite()

    # Add test classes using the recommended approach instead of makeSuite
    test_loader = unittest.TestLoader()
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestClassifierFunctions))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestPerformance))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestUINavigation))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestDatabase))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestPipeline))

    # Run the test suite with custom minimal runner
    runner = MinimalTestRunner(verbosity=1)
    result = runner.run(test_suite)

    # Return non-zero exit code if tests failed
    return len(result.failures) + len(result.errors)


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
