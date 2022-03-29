from fractal_analysis.estimator.hurst_estimator import IrHurstEstimator, QvHurstEstimator
import numpy as np
import math


def test_ir_hurst_estimator_neighborhood():
    estimator = IrHurstEstimator(mbm_series=[1] * 100, alpha=0.2)
    print(estimator._neighborhood(t=99))


def test_ir_hurst_estimator_hest_ir():
    """
        Examples are from the matlab function HestIR(R,p) in the source code:
                    http://samm.univ-paris1.fr/Sofwares-Logiciels
                    Software for estimating the Hurst function H of a Multifractional Brownian Motion:
                    Quadratic Variation estimator and IR estimator
    """
    assert IrHurstEstimator._hest_ir(R=3, p=2) == 0.9999
    assert IrHurstEstimator._hest_ir(R=0.5, p=1) == 0
    assert IrHurstEstimator._hest_ir(R=0.6, p=2) == 0.5797
    assert IrHurstEstimator._hest_ir(R=0.6, p=1) == 0.0500


def test_ir_hurst_estimator_holder_exponents():
    estimator = IrHurstEstimator(mbm_series=np.ones(100), alpha=0.2)
    print(estimator.holder_exponents)

    # Generate a standard brownian motion
    N = 100
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    estimator = IrHurstEstimator(mbm_series=series, alpha=0.2)
    print(estimator.holder_exponents)


def test_qv_hurst_estimator_get_A():
    print(np.round(QvHurstEstimator._get_A(), 4) == np.array([-0.9575, -0.2644, 0.1411, 0.4288, 0.6519]))


def test_qv_hurst_estimator_get_S():
    mbm_series_first_half = np.arange(start=1, stop=50 + 1, step=1)
    mbm_series_second_half = np.arange(start=1, stop=50 + 0.5, step=0.5)
    estimator = QvHurstEstimator(mbm_series=np.concatenate((mbm_series_first_half, mbm_series_second_half)), alpha=0.2)
    print(estimator._get_S(t=70))
    # Get [45.41513761  90.83944954 136.28211009 181.75229358 227.25917431] which is close to
    # the matlab output [45.0023   89.1091  131.1419  171.5714  234.2841]. The difference lies in:
    # (1) function S:  k = self._neighborhood(t=t, step=1) the matlab code uses a neighborhood of every i steps
    # (2) function _neighborhood: the matlab code use fix for ceil and floor.
    print(estimator.holder_exponents[70])  # 0.50023 close to matlab output 0.5013


def test_qv_hurst_estimator_holder_exponents():
    estimator = QvHurstEstimator(mbm_series=np.ones(100), alpha=0.2)
    print(estimator.holder_exponents)

    # Generate a standard brownian motion
    N = 100
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    estimator = QvHurstEstimator(mbm_series=series, alpha=0.2)
    print(estimator.holder_exponents)