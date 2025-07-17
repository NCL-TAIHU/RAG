import unittest
import json
import os

class JSONTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.test_results.append({
            'test': str(test),
            'status': 'success',
            'output': self._stdout_capture.getvalue() if hasattr(self, '_stdout_capture') else ''
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.test_results.append({
            'test': str(test),
            'status': 'failure',
            'error': self._exc_info_to_string(err, test)
        })

    def addError(self, test, err):
        super().addError(test, err)
        self.test_results.append({
            'test': str(test),
            'status': 'error',
            'error': self._exc_info_to_string(err, test)
        })

def main():
    # Discover all tests in the current directory
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__) or '.', pattern='*.py')

    # Collect all test case names
    test_case_names = []
    for test_group in suite:
        for test_case in test_group:
            for test in test_case:
                test_case_names.append(str(test))

    with open('./test_cases.json', 'w') as f:
        json.dump(test_case_names, f, indent=2)

    # Run tests and collect results
    runner = unittest.TextTestRunner(resultclass=JSONTestResult, verbosity=2)
    result = runner.run(suite)

    with open('./test_results.json', 'w') as f:
        json.dump(result.test_results, f, indent=2)

if __name__ == '__main__':
    main() 