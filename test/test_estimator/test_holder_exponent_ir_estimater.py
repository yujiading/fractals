from fractal_analysis.estimator.holder_exponent_ir_estimator import HolderExponentIrEstimator
import numpy as np
import math

def test_holder_exponent_ir_estimator_neighborhood():
    estimator = HolderExponentIrEstimator(mbm_series=[1] * 100, alpha=0.2)
    print(estimator._neighborhood(t=99))


def test_holder_exponent_ir_estimator_hest_ir():
    """
        Examples are from the matlab function HestIR(R,p) in the source code:
                    http://samm.univ-paris1.fr/Sofwares-Logiciels
                    Software for estimating the Hurst function H of a Multifractional Brownian Motion:
                    Quadratic Variation estimator and IR estimator
    """
    assert HolderExponentIrEstimator._hest_ir(R=3, p=2) == 0.9999
    assert HolderExponentIrEstimator._hest_ir(R=0.5, p=1) == 0
    assert HolderExponentIrEstimator._hest_ir(R=0.6, p=2) == 0.5797
    assert HolderExponentIrEstimator._hest_ir(R=0.6, p=1) == 0.0500

def test_holder_exponent_ir_estimator_holder_exponents():
    estimator = HolderExponentIrEstimator(mbm_series=np.ones(100), alpha=0.2)
    print(estimator.holder_exponents)

    # Generate a standard brownian motion
    N=100
    series = np.random.randn(N) * 0.5 * math.sqrt(1 / N)
    series = np.cumsum(series)
    estimator = HolderExponentIrEstimator(mbm_series=series, alpha=0.2)
    print(estimator.holder_exponents)