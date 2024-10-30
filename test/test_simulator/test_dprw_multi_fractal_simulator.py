from fractal_analysis.simulator.dprw.dprw_multi_fractal_simulator import DprwMbmSimulator
import numpy as np


def test_neg_fbm_covariance_func():
    assert round(DprwMbmSimulator.neg_fbm_covariance_func(t=2, s=1, h1=0.2, h2=0.6), 4) == 0.4068


def test_fractal_cov_matrix():
    simulator = DprwMbmSimulator(sample_size=5, holder_exponents=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    print(simulator.fractal_cov_matrix(i=1, hu_=0, hu1_=1, holder_exponent=0.2, means=[0.23, 0.45, 0.75]))
    # matlab output: 0.7643  1.5949  0.1068  -1.4140  -0.8925 -1.1648


def test_riemann_louville_mbm():
    simulator = DprwMbmSimulator(sample_size=3, holder_exponents=np.array([0.1, 0.2, 0.3]))
    print(simulator.riemann_louville_mbm)


def test_get_mbm():
    sample_size = 1000
    t = np.linspace(0, 1, sample_size)
    holder_exponents = 0.9 + 0.01 * np.sin(4 * np.pi * t)
    DprwMbmSimulator(sample_size=sample_size, holder_exponents=holder_exponents).get_mbm(is_plot=True, seed=1,
                                                                                         hurst_name='0.5+0.3sin(4pit)')
