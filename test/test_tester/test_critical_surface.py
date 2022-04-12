from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM, CriticalSurfaceMFBM
import numpy as np
import math


def test_critical_surface_mbm():
    # is_increment_series True vs False
    N = 200
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    add_on_sig = 0
    is_increment_series = True
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=np.ones(N) * 0.5, add_on_sig2=add_on_sig ** 2))
    series = np.diff(series, prepend=0)
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))

    is_increment_series = False
    critical_surface = CriticalSurfaceMFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=np.ones(N) * 0.5, add_on_sig2=add_on_sig ** 2))
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))


def test_critical_surface_fbm():
    # the following test verifies the following sentence in the paper:
    # Michał Balcerek, Krzysztof Burnecki. (2020)
    # Testing of fractional Brownian motion in a noisy environment.
    # Chaos, Solitons & Fractals, Volume 140, 110097.
    # https://doi.org/10.1016/j.chaos.2020.110097

    # "For example, when the analysed data have length N = 200 and we want to check if they come from FBM with noise
    # with H = 0.3 and σ = 0.3, we should look at blue lines in Fig. 3 at σ = 0.3. We read the values −0.51 and −0.16."

    N = 200
    add_on_sig = 0.3
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=True)
    print(critical_surface.quantile(sig2=1, H=0.3, add_on_sig2=add_on_sig ** 2))

    # is_increment_series True vs False
    N = 200
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    add_on_sig = 0
    is_increment_series = True
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=0.5, add_on_sig2=add_on_sig ** 2))
    series = np.diff(series, prepend=0)
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))

    is_increment_series = False
    critical_surface = CriticalSurfaceFBM(N=N, alpha=0.05, is_increment_series=is_increment_series)
    print(critical_surface.quantile(sig2=1, H=0.5, add_on_sig2=add_on_sig ** 2))
    print(np.dot(series.T.dot(critical_surface.matrix_A_k), series))
