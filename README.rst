==================
nose-leak-detector
==================

Mock and memory leak detector plugin for nose.

Features:

* Detects mocks that are not reset between tests.
* Detects mocks created during tests that are not deleted by the end of the test.
* Reports memory delta between tests.  Optional.

============
Django Setup
============

To use this plugin with django nose runner, just include the following in your settings file::

    NOSE_PLUGINS = [
        'nose_leak_detector.LeakDetectorPlugin']
    NOSE_ARGS = ['--with-leak-detector', '--leak-detector-level=3']

