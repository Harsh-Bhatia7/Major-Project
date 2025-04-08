import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import time
import matplotlib
matplotlib.use('Agg')  # Disable plots from showing during tests


class MinimalTestResult(unittest.TextTestResult):
    """A test result class that logs minimal information about test runs."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.test_logs = {}
        self.successes = []
        self.start_time = None

    def startTest(self, test):
        self.start_time = time.time()
        super().startTest(test)

        # Capture stdout and stderr during the test
        self.captured_output = StringIO()
        self.stdout_redirect = redirect_stdout(self.captured_output)
        self.stderr_redirect = redirect_stderr(self.captured_output)
        self.stdout_redirect.__enter__()
        self.stderr_redirect.__enter__()

    def stopTest(self, test):
        # Stop capturing stdout and stderr
        self.stdout_redirect.__exit__(None, None, None)
        self.stderr_redirect.__exit__(None, None, None)
        output = self.captured_output.getvalue()

        # Store captured output
        test_name = self.getDescription(test)
        test_class = test.__class__.__name__
        if test_class not in self.test_logs:
            self.test_logs[test_class] = {}

        duration = time.time() - self.start_time
        self.test_logs[test_class][test_name] = {
            'output': output,
            'duration': duration
        }

        super().stopTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        self.stream.write('.')
        self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write('E')
        self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write('F')
        self.stream.flush()

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.stream.write('S')
        self.stream.flush()

    def printErrors(self):
        self.stream.writeln()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)


class MinimalTestRunner(unittest.TextTestRunner):
    """A test runner that displays results in a minimal format."""

    resultclass = MinimalTestResult

    def run(self, test):
        """Run the given test case or test suite."""
        result = super().run(test)
        self.stream.writeln()
        self.stream.writeln("TEST SUMMARY")
        self.stream.writeln("=" * 40)

        # Print summary of test classes
        for test_class, tests in result.test_logs.items():
            self.stream.writeln(f"\n{test_class}:")
            for test_name, data in tests.items():
                # Extract just the method name from the test_name
                test_short_name = test_name.split(' ')[1]  # Get just the method name
                duration = data['duration']

                # Use unittest's TestCase._id to match tests
                status = "✅ PASS" if any(test_name == result.getDescription(t) for t in result.successes) else "❌ FAIL"

                self.stream.writeln(f"  {status} {test_short_name} ({duration:.3f}s)")

        # Print overall summary
        self.stream.writeln("\nSUMMARY")
        self.stream.writeln("-" * 40)
        self.stream.writeln(f"Ran {result.testsRun} tests")
        self.stream.writeln(f"Successes: {len(result.successes)}")
        self.stream.writeln(f"Failures: {len(result.failures)}")
        self.stream.writeln(f"Errors: {len(result.errors)}")

        return result
