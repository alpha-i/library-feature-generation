
- [x] Finish my changes
- [ ] Push my commits to GitHub
- [ ] Open a pull request


## Setup Development Environment

### Create conda environment
```bash
$ conda create -n feat-gen-env python=3.5
$ source activate feat-gen-env
```

### Install dependencies

```bash
$ pip install -U setuptools --ignore-installed --no-cache-dir
$ pip install -r dev-requirements.txt --src $CONDA_PREFIX
```

### Running the test suite
```bash
pytest tests/
```


