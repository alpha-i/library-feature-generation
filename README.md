
Features we want to implement 

- [ ] Fourier transform / power spectrum 
- [ ] Wavelet 
- [ ] Kalmann Filter
- [x] Exponentail moving average  
- [ ] Commodity channel index: helps to find the start and the end of a trend.
- [ ] ATR Average true range: measures the volatility of price.
- [ ] Moving average 
- [ ] N-day momentum Momentum: helps pinpoint the end of a decline or advance
- [ ] ROC Price rate of change: shows the speed at which a stock’s price is changing
- [ ] Stochastic Momentum Index: shows where the close price is relative to the midpoint of the same range.
 -[ ] WVAD Williams’s Variable Accumulation/Distribution: measures the buying and selling pressure



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


