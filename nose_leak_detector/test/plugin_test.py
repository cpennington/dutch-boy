from builtins import *

import StringIO
import contextlib
import mock
import re
import unittest

from nose_leak_detector import plugin

@contextlib.contextmanager
def _leaked_mock(case):
    """ Create a leaked mock for the duration of a context """
    leaked_mock = mock.Mock(name='%s: MY LEAKED MOCK' % case.id())
    yield
    del leaked_mock


class LeakDetectorFinalizeTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = plugin.LeakDetectorPlugin()
        options = mock.Mock(
            name='%s: Options NOSE_LEAK_DETECTOR_IGNORE' % self.id(),
            leak_detector_level=plugin.LEVEL_TEST,
            leak_detector_report_delta=False,
            leak_detector_patch_mock=True)

        configuration = mock.Mock(name='%s: Configuration NOSE_LEAK_DETECTOR_IGNORE' % self.id())
        self.detector.configure(options, configuration)
        self.detector.begin()

        # detector requires a test to attach a failure to
        self.fake_test = mock.MagicMock(name='%s: Test NOSE_LEAK_DETECTOR_IGNORE' % self.id())
        self.fake_result = mock.create_autospec(unittest.TestResult,
                                                _name='%s: Result NOSE_LEAK_DETECTOR_IGNORE'
                                                      % self.id())

        # Simulate running a nose test
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)
        self.detector.afterTest(self.fake_test)

        self.suite_result = mock.create_autospec(unittest.result.TestResult,
                                                 _name='Suite Result NOSE_LEAK_DETECTOR_IGNORE')
        self.suite_result.errors = []

    def tearDown(self):
        del self.suite_result
        del self.fake_test
        del self.fake_result
        del self.detector

    def test_leak_detected(self):
        """ Leaks are reported as errors when the test suite finishes. """

        with _leaked_mock(self):
            stream = StringIO.StringIO()
            self.detector.report(stream)
            self.assertRegexpMatches(stream.getvalue(),
                                     re.compile('FAILED.*Found 1 new mock.*MY LEAKED MOCK',
                                                re.MULTILINE | re.DOTALL))

        self.detector.finalize(self.suite_result)
        self.assertTrue(self.suite_result.addError.called)
        self.assertEquals(self.suite_result.addError.call_args[0][0], self.fake_test.test)

    def test_no_leak_detected(self):
        """ No leak is detected in normal test case. """
        stream = StringIO.StringIO()
        self.detector.report(stream)
        self.assertRegexpMatches(stream.getvalue(), '.*PASSED.*')

        self.detector.finalize(self.suite_result)
        self.assertFalse(self.suite_result.addError.called)


class LeakDetectorLevelTestCase(unittest.TestCase):
    def setUp(self):
        self.detector = plugin.LeakDetectorPlugin()
        options = mock.Mock(
            name='%s: Options NOSE_LEAK_DETECTOR_IGNORE' % self.id(),
            leak_detector_level=plugin.LEVEL_TEST,
            leak_detector_report_delta=False,
            leak_detector_patch_mock=True)

        configuration = mock.Mock(name='%s: Configuration NOSE_LEAK_DETECTOR_IGNORE' % self.id())
        self.detector.configure(options, configuration)
        self.detector.begin()
        self.fake_test = mock.MagicMock(name='%s: Test NOSE_LEAK_DETECTOR_IGNORE'
                                             % self.id())
        self.fake_result = mock.create_autospec(unittest.result.TestResult,
                                                _name='%s: Result NOSE_LEAK_DETECTOR_IGNORE'
                                                      % self.id())

    def tearDown(self):
        del self.detector
        del self.fake_test
        del self.fake_result

    def test_leak_detected(self):
        """ A mock leaked by one test is detected on the next test but reported on the first. """

        # Fake one test running
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)

        with _leaked_mock(self):
            self.detector.afterTest(self.fake_test)

            next_test = mock.Mock(name='%s: Next Test NOSE_LEAK_DETECTOR_IGNORE' % self.id())
            next_result = mock.create_autospec(unittest.result.TestResult,
                                               _name='%s: Next Result NOSE_LEAK_DETECTOR_IGNORE'
                                                     % self.id())

            # Simulate nose running another test
            self.detector.beforeTest(next_test)
            self.detector.prepareTestCase(next_test)(next_result)

        self.detector.afterTest(next_test)

        # The error should be set on the first test
        self.assertTrue(self.fake_result.addError.called)
        self.assertEquals(self.fake_result.addError.call_args[0][0], self.fake_test)
        self.assertRegexpMatches(str(self.fake_result.addError.call_args[0][1]),
                                 re.compile('Found 1 new mock.*MY LEAKED MOCK',
                                            re.MULTILINE | re.DOTALL))

    def test_no_leak_detected(self):
        """ No errors should be generated when there are no mocks present prior to a test. """

        # Simulate nose running a test
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)
        self.detector.afterTest(self.fake_test)

        self.assertFalse(self.fake_result.addError.called)

    def test_leak_detected_before_first_test(self):
        """ Mocks that exist before the first test are reported as errors on the first test. """

        with _leaked_mock(self):
            # Simulate nose running a test
            self.detector.beforeTest(self.fake_test)
            prepared_test = self.detector.prepareTestCase(self.fake_test)
            prepared_test(self.fake_result)

        self.detector.afterTest(self.fake_test)

        self.assertTrue(self.fake_result.addError.called)
        self.assertEquals(self.fake_result.addError.call_count, 1)
        self.assertEquals(self.fake_result.addError.call_args[0][0], self.fake_test)
        self.assertRegexpMatches(str(self.fake_result.addError.call_args[0][1]),
                                 re.compile('Found 1 new mock.*MY LEAKED MOCK',
                                            re.MULTILINE | re.DOTALL))
