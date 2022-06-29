# Fractal Analysis
Fractal and multifractal methods, including 

- fractional Brownian motion (FBM) tester
- multifractional Brownian motion (MBM) tester
- IR hurst exponents estimator of multifractional Brownian motion (MBM)
- QV hurst exponents estimator of multifractional Brownian motion (MBM)

## FBM / MBM tester
Test if a series is FBM (MBM) given the hurst parameter (hurst exponents series).
The implementation is based on the following papers:

>Michał Balcerek, Krzysztof Burnecki. (2020)  
Testing of fractional Brownian motion in a noisy environment.  
Chaos, Solitons & Fractals, Volume 140, 110097.  
https://doi.org/10.1016/j.chaos.2020.110097

>Balcerek, Michał, and Krzysztof Burnecki. (2020)  
Testing of Multifractional Brownian Motion. Entropy 22, no. 12: 1403.  
https://doi.org/10.3390/e22121403 

We added the following improvements to the FBM and/or MBM tester:
- option for automatically estimating sigma 
  - based on Theorem 2.3 of the following paper:
    >Ayache A., Peng Q. (2012)  
     Stochastic Volatility and Multifractional Brownian Motion.  
     In: Zili M., Filatova D. (eds) Stochastic Differential Equations and Processes. 
     Springer Proceedings in Mathematics, vol 7. Springer, Berlin, Heidelberg.  
     https://doi.org/10.1007/978-3-642-22368-6_6
  - a detailed introduction can be found in the section 5 of the following paper:
    > todo: add paper name
- option for testing if the series itself is a FBM (MBM)
- option for testing if the increment of the series is the increment of a FBM (MBM)
- option for testing if the series is a FBM (MBM) with an add-on noise
- option for testing if the increment of the series is the increment of a FBM (MBM) with an add-on noise

## IR / QV hurst estimator of MBM
Estimate the hurst parameter (hurst exponent series) of a MBM.
The implementation is based on the following paper:

>Bardet, Jean-Marc & Surgailis, Donatas, 2013.  
Nonparametric estimation of the local Hurst function of multifractional Gaussian processes.  
Stochastic Processes and their Applications, Elsevier, vol. 123(3), pages 1004-1045.


Bardet, the author in the above paper, provides a Matlab code that can be found at: 
>http://samm.univ-paris1.fr/Sofwares-Logiciels  
Software for estimating the Hurst function H of a Multifractional Brownian Motion:
 Quadratic Variation estimator and IR estimator


## To install
To get started, simply do:
```
pip install fractal-analysis
```
or check out the code from out GitHub repository.

You can now use the series tester module in Python with:
```
from fractal_analysis import tester
```
or use the hurst estimator with
```
from fractal_analysis import estimator
```

## Examples
### FBM / MBM tester
Import:
```
from fractal_analysis.tester.series_tester import MBMSeriesTester, FBMSeriesTester
from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM, CriticalSurfaceMFBM
```
To test if a series ```series``` is FBM, one needs to use ```CriticalSurfaceFBM``` with length of the series ```N```,
the significance level ```alpha``` (look at quantiles of order ```alpha/2``` and ```1 − alpha/2```), and  choose to test
on the series itself or its increment series using ```is_increment_series``` (default is ```False```, meaning to test on
the series itself),
```
fbm_tester = FBMSeriesTester(critical_surface=CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=False))
```

To test if the series is FBM with hurst parameter 0.3 and use auto estimated sigma square (set ```sig2=None```):

```
is_fbm, sig2 = fbm_tester.test(h=0.3, x=series, sig2=None, add_on_sig2=0)
```
If the output contains, for example:
> Bad auto sigma square calculated with error 6.239236333681868. Suggest to give sigma square and rerun.

The auto sigma square estimated is not accurate. One may want to manually choose a sigma square and rerun. For example:
```
is_fbm, sig2 = fbm_tester.test(h=0.3, x=series, sig2=1, add_on_sig2=0)
```
If one wants to test with an add-no noise, change the value of ```add_on_sig2```.




To test if the series is MBM, one needs to use ```CriticalSurfaceMFBM``` with length of the series ```N```
and the significance level ```alpha``` (look at quantiles of order ```alpha/2``` and ```1 − alpha/2```) 
```
mbm_tester = MBMSeriesTester(critical_surface=CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=False))
```
To test if the series is MBM with a given holder exponent series ```h_mbm_series``` and use auto estimated sigma square:
```
is_mbm, sig2 = mbm_tester.test(h=h_mbm_series, x=series, sig2=None, add_on_sig2=0)
```
Be aware that ```MBMSeriesTester``` requires ```len(h_mbm_series)==len(series)```.

#### Use of cache
Use caching to speed up the testing process. If the series ```x``` for testing is unchanged and multiple ```h``` 
and/or ```sig2``` are used, one may want to set 
```is_cache_stat=True``` to allow cache variable ```stat```. If ```h``` and ```sig2``` are unchanged and multiple ```x```
are used, one may want to set ```is_cache_quantile=True``` to allow cache variable ```quantile```. For example:
```
mbm_tester = MBMSeriesTester(critical_surface=CriticalSurfaceMFBM(N=N, alpha=0.05), is_cache_stat=True, is_cache_quantile=False)
```

### IR / QV hurst estimator of MBM
Import:
```
from fractal_analysis.estimator.hurst_estimator import IrHurstEstimator, QvHurstEstimator
import numpy as np
import math
```
Generate a standard brownian motion
```
N = 100
series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
series = np.cumsum(series)
```
To estimate the hurst exponents series of the above series with ```alpha=0.2``` using IR estimator,
```
estimator = IrHurstEstimator(mbm_series=series, alpha=0.2)
print(estimator.holder_exponents)
```
To estimate the hurst exponents series of the above series with ```alpha=0.2``` using QV estimator,
```
estimator = QvHurstEstimator(mbm_series=series, alpha=0.2)
print(estimator.holder_exponents)
```
Here the value of ```alpha``` decides how many observations on the ```mbm_series``` is used to estimate a point of the
holder exponent; small alpha means more observations are used for a single point and therefore the variance is small.
   
