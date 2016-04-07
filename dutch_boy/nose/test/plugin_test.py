from builtins import *

from io import StringIO
import contextlib
try:
    from unittest import mock
except ImportError:
    import mock
import re
import unittest

import dutch_boy.nose


class TestCase(unittest.TestCase):
    pass

# For python 2.7
if not hasattr(TestCase, 'assertRegex'):
    TestCase.assertRegex = TestCase.assertRegexpMatches
    TestCase.assertRaisesRegex = TestCase.assertRaisesRegexp

@contextlib.contextmanager
def _leaked_mock(case, name='MY LEAKED MOCK'):
    """ Create a leaked mock for the duration of a context """
    leaked_mock = mock.Mock(name='%s: %s' % (case.id(), name))
    yield
    del leaked_mock


def create_ignored_mock(name='', **kwargs):
    """ Create a mock that won't be considered leaked by the leak detector """
    name += ' NOSE_LEAK_DETECTOR_IGNORE'
    return mock.MagicMock(name=name, **kwargs)


class LeakDetectorFinalizeTestCase(TestCase):
    def setUp(self):
        self.detector = dutch_boy.nose.LeakDetectorPlugin()
        options = create_ignored_mock(
            name='%s: Options' % self.id(),
            leak_detector_level=dutch_boy.nose.plugin.LEVEL_TEST,
            leak_detector_report_delta=False,
            leak_detector_patch_mock=True,
            leak_detector_save_traceback=False,
            leak_detector_ignore_patterns=['NOSE_LEAK_DETECTOR_IGNORE'])

        configuration = create_ignored_mock(name='%s: Configuration' % self.id())
        self.detector.configure(options, configuration)
        self.pre_existing_mock = mock.Mock(name='%s: %s' % (self.id(), 'MY PRE EXISTING MOCK'))
        self.detector.begin()

        # detector requires a test to attach a failure to
        self.fake_test = create_ignored_mock(name='%s: Test' % self.id(),
                                             __class__=create_ignored_mock(__name__='testClass'),
                                             __unicode__=create_ignored_mock(return_value='my_test'))
        self.fake_result = create_ignored_mock(spec=unittest.TestResult,
                                               name='%s: Result' % self.id())

        # Simulate running a nose test
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)
        self.detector.afterTest(self.fake_test)

        self.suite_result = create_ignored_mock(spec=unittest.result.TestResult,
                                                name='Suite Result')
        self.suite_result.errors = []

    def tearDown(self):
        # Nose keeps tests are around (in this case via the _precache attribute of ContextSuite)
        # so we have to ensure this is removed
        del self.pre_existing_mock

    def test_leak_detected(self):
        """ Leaks are reported as errors when the test suite finishes. """

        with _leaked_mock(self), StringIO() as stream:
            self.detector.report(stream)
            self.assertRegex(stream.getvalue(),
                             re.compile('FAILED.*Found 1 new mock.*MY LEAKED MOCK',
                                        re.MULTILINE | re.DOTALL))

        with self.assertRaisesRegex(dutch_boy.nose.plugin.LeakDetected,
                                    'Found 1 new mock'):
            self.detector.finalize(self.suite_result)

    def test_pre_existing_mock_called_and_not_reset(self):
        """ Mocks that have been called and not reset are reported. """

        # Call the existing mock
        self.pre_existing_mock()
        with StringIO() as stream:
            self.detector.report(stream)
            self.assertRegex(stream.getvalue(),
                             re.compile('FAILED.*Found 1 dirty mock.*MY PRE EXISTING MOCK',
                                        re.MULTILINE | re.DOTALL))

        with self.assertRaisesRegex(dutch_boy.nose.plugin.LeakDetected,
                                    'Found 1 dirty mock'):
            self.detector.finalize(self.suite_result)

    def test_no_leak_detected(self):
        """ No leak is detected in normal test case. """
        with StringIO() as stream:
            self.detector.report(stream)
            self.assertRegex(stream.getvalue(), '.*PASSED.*')

        self.detector.finalize(self.suite_result)
        self.assertFalse(self.suite_result.addError.called)


class LeakDetectorLevelTestCase(TestCase):
    def setUp(self):
        self.detector = dutch_boy.nose.LeakDetectorPlugin()
        options = create_ignored_mock(
            name='%s: Options' % self.id(),
            leak_detector_level=dutch_boy.nose.plugin.LEVEL_TEST,
            leak_detector_report_delta=False,
            leak_detector_patch_mock=True,
            leak_detector_save_traceback=False,
            leak_detector_ignore_patterns=['NOSE_LEAK_DETECTOR_IGNORE'])

        configuration = create_ignored_mock(name='%s: Configuration' % self.id())
        self.detector.configure(options, configuration)
        self.pre_existing_mock = mock.Mock(name='%s: %s' % (self.id(), 'MY PRE EXISTING MOCK'))
        self.detector.begin()
        self.fake_test = create_ignored_mock(name='%s' % self.id(),
                                             __class__=create_ignored_mock(__name__='testClass'),
                                             __unicode__=create_ignored_mock(return_value='my_test'))
        self.fake_result = create_ignored_mock(spec=unittest.result.TestResult,
                                               name='%s: Result' % self.id())

    def test_leak_detected(self):
        """ A mock leaked by one test is detected on the next test but reported on the first. """

        # Fake one test running
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)

        with _leaked_mock(self):
            self.detector.afterTest(self.fake_test)

            next_test = create_ignored_mock(name='%s: Next Test' % self.id())
            next_result = create_ignored_mock(spec=unittest.result.TestResult,
                                              name='%s: Next Result' % self.id())

            # Simulate nose running another test
            self.detector.beforeTest(next_test)
            self.detector.prepareTestCase(next_test)(next_result)

        self.detector.afterTest(next_test)

        # The error should be set on the second test (because we don't want to keep a test around)
        self.assertTrue(next_result.addError.called)
        self.assertEqual(next_result.addError.call_args[0][0], next_test)
        self.assertRegex(str(next_result.addError.call_args[0][1]),
                         re.compile('Found 1 new mock.*MY LEAKED MOCK',
                                    re.MULTILINE | re.DOTALL))

    def test_pre_existing_mock_called_and_not_reset(self):
        """ Existing that have been called and not reset are reported. """

        # Fake one test running
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)

        # Call the pre-existing mock
        self.pre_existing_mock()
        self.detector.afterTest(self.fake_test)

        next_test = create_ignored_mock(name='%s: Next Test' % self.id())
        next_result = create_ignored_mock(spec=unittest.result.TestResult,
                                          name='%s: Next Result' % self.id())

        # Simulate nose running another test
        self.detector.beforeTest(next_test)
        self.detector.prepareTestCase(next_test)(next_result)
        self.detector.afterTest(next_test)

        # The error should be set on the second test (because we don't want to keep a test around)
        self.assertTrue(next_result.addError.called)
        self.assertEqual(next_result.addError.call_args[0][0], next_test)
        self.assertRegex(str(next_result.addError.call_args[0][1]),
                         re.compile('Found 1 dirty mock.*MY PRE EXISTING MOCK',
                                    re.MULTILINE | re.DOTALL))

    def test_no_leak_detected(self):
        """ No errors should be generated when there are no mocks present prior to a test. """

        # Simulate nose running a test
        self.detector.beforeTest(self.fake_test)
        self.detector.prepareTestCase(self.fake_test)(self.fake_result)
        self.detector.afterTest(self.fake_test)
        self.assertFalse(self.fake_result.addError.called)

    def tearDown(self):
        # Nose keeps tests are around (in this case via the _precache attribute of ContextSuite)
        # so we have to ensure this is removed
        del self.pre_existing_mock
