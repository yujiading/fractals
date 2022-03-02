import pprint as pp
from math import sqrt

import numpy as np

from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM, CriticalSurfaceMFBM
from fractal_analysis.tester.series_tester import FBMSeriesTester, MBMSeriesTester


def test_tester_fbm_constant():
    # Set series length
    N = 100
    # Generate a constant series
    series = np.ones(N)
    # Test if it is a FBM. Alpha is the significance level (look at quantiles of order alpha/2 and 1 − alpha/2)
    fbm_tester = FBMSeriesTester(critical_surface=CriticalSurfaceFBM(N=N, alpha=0.05))
    # Use holder exponent 0.9 and auto estimated sigma square (set sig2=None)
    is_fbm, sig2 = fbm_tester.test(h=0.9, x=series, sig2=None)
    # Print the result
    pp.pprint(f"is fbm: {is_fbm} | sigma2: {sig2}")

def test_tester_fbm_standard_brownian_motion():
    # Set series length
    N = 100
    # Generate a standard brownian motion
    series = np.random.randn(N) * 0.5 * sqrt(1 / N)
    series = np.cumsum(series)
    # Test if it is a FBM. Alpha is the significance level (look at quantiles of order alpha/2 and 1 − alpha/2)
    fbm_tester = FBMSeriesTester(critical_surface=CriticalSurfaceFBM(N=N, alpha=0.05, k=1))
    # Use holder exponent 0.5 and auto estimated sigma square (set sig2=None)
    ret, sig2 = fbm_tester.test(h=0.5, x=series, sig2=None)
    # Print the result
    pp.pprint(f"is fbm: {ret} | sigma2: {sig2}")

def test_tester_mbm_standard_brownian_motion():
    # Set series length
    N = 100
    # Generate a standard brownian motion
    series = np.random.randn(N) * 0.5 * sqrt(1 / N)
    series = np.cumsum(series)
    # Test if it is a FBM. Alpha is the significance level (look at quantiles of order alpha/2 and 1 − alpha/2)
    mbm_tester = MBMSeriesTester(critical_surface=CriticalSurfaceMFBM(N=N, alpha=0.05))

    # # Use holder exponent 0.5 and auto estimated sigma square (set sig2=None)
    # is_mbm, sig2 = mbm_tester.test(h=0.5, x=series, sig2=None)
    # # Return error message: ValueError: h must be array-like for MBM tester.
    # # Use a constant (0.5) holder exponent series and auto estimated sigma square (set sig2=None)
    # is_mbm, sig2 = mbm_tester.test(h=[0.5, 0.5], x=series, sig2=None)
    # # Return error message: ValueError: h and x should have the same length for MBM tester.

    # Use a constant (0.5) holder exponent series and use 0.001 sigma square
    is_mbm, sig2 = mbm_tester.test(h=np.ones(N)*0.5, x=series, sig2=0.001)
    # Print the result
    pp.pprint(f"is mbm: {is_mbm} | sigma2: {sig2}")

def test_tester_fbm_constant_with_cache():
    # Set series length
    N = 100
    # Generate a constant series
    series = np.ones(N)
    # Test if it is a FBM. Alpha is the significance level (look at quantiles of order alpha/2 and 1 − alpha/2)
    fbm_tester = FBMSeriesTester(critical_surface=CriticalSurfaceFBM(N=N, alpha=0.05), is_cache_stat=False, is_cache_quantile=True)
    # Use holder exponent 0.9 and auto estimated sigma square (set sig2=None)
    is_fbm, sig2 = fbm_tester.test(h=0.9, x=series, sig2=None)
    # Print the result
    pp.pprint(f"is fbm: {is_fbm} | sigma2: {sig2}")
    # Use holder exponent 0.9 and auto estimated sigma square (set sig2=None)
    is_fbm, sig2 = fbm_tester.test(h=0.9, x=series, sig2=None)
    # Print the result
    pp.pprint(f"is fbm: {is_fbm} | sigma2: {sig2}")
    # Use holder exponent 0.9 and auto estimated sigma square (set sig2=None)
    is_fbm, sig2 = fbm_tester.test(h=0.9, x=series, sig2=None)
    # Print the result
    pp.pprint(f"is fbm: {is_fbm} | sigma2: {sig2}")

