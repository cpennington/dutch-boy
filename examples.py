""" Some examples of what nose leak detector can detect. """

try:
    from unittest import mock
except ImportError:
    from mock import mock

new_global_mock = None


def test_with_leaked_new_global_mock():
    global new_global_mock
    new_global_mock = mock.Mock()


called_global_mock = mock.Mock()


def test_with_leaked_called_global_mock():
    global called_global_mock
    called_global_mock()
