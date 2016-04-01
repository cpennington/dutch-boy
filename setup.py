import sys
import codecs

# Prevent spurious errors during `python setup.py test`, a la
# http://www.eby-sarna.com/pipermail/peak/2010-May/003357.html:
try:
    import multiprocessing
except ImportError:
    pass

from setuptools import setup, find_packages


extra_setup = {}
if sys.version_info >= (3,):
    extra_setup['use_2to3'] = True

setup(
    name='nose-leak-detector',
    version='0.1.5',
    description='A Nose plugin that detects leaks between tests, mainly with mocks',
    long_description=codecs.open('README.rst', encoding='utf-8').read(),
    keywords=['testing', 'memory', 'nose', 'mocks'],
    author='Andrew S. Brown',
    author_email='asbrown@nextdoor.com',
    license='BSD',
    packages=find_packages(exclude=['ez_setup']),
    install_requires=[
        'future>=0.15.2',
        'nose>=1.2.1',
        'pympler>=0.3.1',
        'termcolor>=1.1.0',
    ],
    tests_require=[
        'mock>=1.3.0',
    ],
    test_suite='nose.collector',
    url='https://github.com/Nextdoor/nose-leak-detector',
    download_url='https://github.com/Nextdoor/nose-leak-detector/archive/0.1.5'
                 '#egg=nose-leak-detector-0.1.5',
    include_package_data=True,
    entry_points="""
        [nose.plugins.0.10]
        nose_leak_detector = nose_leak_detector.plugin:LeakDetectorPlugin
        """,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Topic :: Software Development :: Testing'
        ],
    **extra_setup
)
