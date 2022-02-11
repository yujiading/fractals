import pprint as pp

import pandas as pd
import numpy as np
from tester.library.series_tester import SeriesTester


def test_tester():
    # series = np.random.randn(50)
    series = np.ones(50)
    tester = SeriesTester(x=series)
    ## test if fbm
    # given h, auto sig2
    h_fbm = 1
    ret, sig2, h = tester.is_fbm(h=h_fbm, sig2=None)
    pp.pprint(f"is fbm: {ret} | sigma2: {sig2} | h: {h}")
    # try all h, sig2=1
    ret, sig2, h = tester.is_fbm(h=None, sig2=1)
    pp.pprint(f"is fbm: {ret} | sigma2: {sig2} | h: {h}")

    ## test if mbm
    h_mbm = np.random.uniform(0, 1, 50)
    # auto sig2
    ret, sig2 = tester.is_mbm(h=h_mbm, sig2=None)
    pp.pprint(f"is mbm: {ret} | sigma2: {sig2}")
    # sig2=1
    ret, sig2 = tester.is_mbm(h=h_mbm, sig2=1)
    pp.pprint(f"is mbm: {ret} | sigma2: {sig2}")