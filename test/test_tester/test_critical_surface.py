from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM, CriticalSurfaceMFBM
import numpy as np
import math
import pandas as pd


def test_add_on_sig2_matrix():
    N = 5
    add_on_sig = 0.3
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=True)
    print(critical_surface._add_on_sig2_matrix(add_on_sig2=add_on_sig ** 2))


def test_matrix_A_k():
    N = 5
    critical_surface = CriticalSurfaceMFBM(N=N, k=1)
    print(critical_surface.matrix_A_k * (N - 1))


def test_mbm_covariance():
    N = 5
    add_on_sig = 0.3
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=True)
    print(critical_surface._autocovariance_matrix(sig2=1.2, H=np.ones(N) * 0.3, add_on_sig2=add_on_sig ** 2))
    print(critical_surface._autocovariance_matrix_increment(sig2=1, H=np.ones(N) * 0.3, add_on_sig2=add_on_sig ** 2))


def test_fbm_covariance():
    N = 5
    add_on_sig = 0.3
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=True)
    print(critical_surface._autocovariance_matrix(sig2=1.2, H=0.3, add_on_sig2=add_on_sig ** 2))
    print(critical_surface._autocovariance_matrix_increment(sig2=1, H=0.3, add_on_sig2=add_on_sig ** 2))


def test_critical_surface_mbm():
    # is_increment_series True vs False
    N = 200
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    add_on_sig = 0
    is_increment_series = True
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=np.ones(N) * 0.5, add_on_sig2=add_on_sig ** 2))
    series_inc = np.diff(series, prepend=0)
    print(np.dot(series_inc.T.dot(critical_surface.matrix_A_k), series_inc))

    is_increment_series = False
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=np.ones(N) * 0.5, add_on_sig2=add_on_sig ** 2))
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))


def test_critical_surface_fbm():
    # is_increment_series True vs False
    N = 200
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    add_on_sig = 0
    is_increment_series = True
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=0.5, add_on_sig2=add_on_sig ** 2))
    series_inc = np.diff(series, prepend=0)
    print(np.dot(series_inc.T.dot(critical_surface.matrix_A_k), series_inc))

    is_increment_series = False
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=0.5, add_on_sig2=add_on_sig ** 2))
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))


def test_real_date_mbm():
    trial = 17
    is_increment_series = True
    add_on_sig = 0
    sig2 = 1
    alpha = 0.01
    is_mean = False # useless when use increment
    is_std = False

    # series_df = pd.read_csv('woodchan_normal_case.csv', header=None)
    series_df = pd.read_csv('dpr_lower_case.csv', header=None)
    N = series_df.shape[0]
    series = series_df.iloc[:, trial]
    if is_mean:
        series = series - series.mean()
    if is_std:
        series = series / series.std()
    t = np.linspace(0, 1, N, endpoint=True)
    # h = 0.5 + 0.3 * np.sin(4 * np.pi * t)
    h = 0.1 + 0.01 * np.sin(4 * np.pi * t)
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=alpha, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=sig2, H=h, add_on_sig2=add_on_sig ** 2))
    if is_increment_series:
        series_inc = np.diff(series, prepend=0)
        print(np.dot(series_inc.T.dot(critical_surface.matrix_A_k), series_inc))
    else:
        print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))

def test_real_date_fbm():
    trial = 17
    is_increment_series = True
    add_on_sig = 0
    sig2 = 1
    alpha = 0.01
    is_mean = False # useless when use increment
    is_std = False

    series_df = pd.read_csv('woodchan_fBm_0.2.csv', header=None)
    N = series_df.shape[0]
    series = series_df.iloc[:, trial]
    if is_mean:
        series = series - series.mean()
    if is_std:
        series = series / series.std()
    t = np.linspace(0, 1, N, endpoint=True)
    h = 0.2
    critical_surface = CriticalSurfaceFBM(N=N, alpha=alpha, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=sig2, H=h, add_on_sig2=add_on_sig ** 2))
    if is_increment_series:
        series_inc = np.diff(series, prepend=0)
        print(np.dot(series_inc.T.dot(critical_surface.matrix_A_k), series_inc))
    else:
        print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))
