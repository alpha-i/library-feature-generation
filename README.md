# library-feature-generation
generate time series features 

```
Setup Development Environment
-----------------------------

###Create conda environment
```bash
$ conda create -n feat-gen-env python=3.5
$ source activate feat-gen-env
```
###Install dependencies
```bash
$ pip  install --upgrade --ignore-installed --no-cache-dir setuptools pip
$ pip install -r dev-requirements.txt 

###Running the test suite
```bash
$ PYTHONPATH=. pytest tests/
```

Install
=======

To install run
```text
$ python setup.py install
```

