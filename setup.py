from setuptools import setup
from setuptools import find_packages

setup(
    name='alphai_feature_generation',
    version='2.0.0',
    description='Alpha-I Feature Generation',
    author='Fergus Simpson',
    author_email='fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=['pandas<0.22',
                      'scikit-learn',
                      'numpy',
                      'scipy',
                      'alphai_calendars>=0.0.1,<1.0.0',
                      'tables',
                      'pyts',
                      'future',
                      'python-dateutil',
                  ],
    dependency_links=[
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_calendars/',
    ]
)
