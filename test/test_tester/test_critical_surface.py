from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM, CriticalSurfaceMFBM
import numpy as np


def test_critical_surface_mbm():
    critical_surface = CriticalSurfaceMFBM(N=1000, alpha=0.05)
    print(critical_surface._generalized_chi2(sig2=1, H=0.3 * np.arange(1000) / 1000 + 0.3, add_on_sig2=0))
    # print(critical_surface.quantile(sig2=1, H=0.3 * np.arange(1000) / 1000 + 0.3))
