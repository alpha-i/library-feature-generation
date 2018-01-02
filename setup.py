from setuptools import setup
from setuptools import find_packages

setup(
    name='alphai_feature_generation',
    version='0.0.1',
    description='Alpha-I Feature Generation',
    author='Christopher Bonnett',
    author_email='christopher.bonnett@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=['numpy',
                      'pandas',
                      'scikit-learn',
                      'scipy',
                      'pandas_market_calendars=>0.12',
                      'tables',
                      'pyts',
                      'future',
                      'python-dateutil'
                      ]
)
