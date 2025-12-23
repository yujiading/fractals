from fractal_analysis.simulator.dpw.dpw_fractal_simulator import DpwSubFbmSimulator, DpwFbmSimulator, \
    DpwNegFbmSimulator, DpwBiFbmSimulator, DpwTriFbmSimulator


def test_lamperti_subseq_index():
    simulator = DpwSubFbmSimulator(sample_size=1024, hurst_parameter=0.1, tmax=15)
    print(simulator._lamperti_subseq_index)


def test_first_line_circulant_matrix_sub_fbm():
    dpw_sub_fbm_simulator = DpwSubFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dpw_sub_fbm_simulator._first_line_circulant_matrix(m=20, cov=dpw_sub_fbm_simulator.covariance_line))


def test_dpw_sub_fbm_simulator():
    sub_fbm = DpwSubFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_sub_fbm(is_plot=True, seed=1)
    sub_fbm = DpwSubFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_sub_fbm(is_plot=True, seed=1)


def test_dpw_bi_fbm_simulator():
    bi_fbm = DpwBiFbmSimulator(sample_size=1000, hurst_parameter=0.8, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DpwBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DpwBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=1).get_bi_fbm(is_plot=True, seed=1)


def test_dpw_tri_fbm_simulator():
    tri_fbm = DpwTriFbmSimulator(sample_size=1000, hurst_parameter=0.8, tri_factor=0.2).get_tri_fbm(is_plot=True,
                                                                                                     seed=1)
    tri_fbm = DpwTriFbmSimulator(sample_size=1000, hurst_parameter=0.2, tri_factor=0.2).get_tri_fbm(is_plot=True,
                                                                                                     seed=1)
    tri_fbm = DpwTriFbmSimulator(sample_size=1000, hurst_parameter=0.2, tri_factor=1).get_tri_fbm(is_plot=True, seed=1)


def test_dpw_fbm_simulator():
    fbm = DpwFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_fbm(is_plot=True, seed=1)
    fbm = DpwFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_fbm(is_plot=True, seed=1)
    fbm = DpwFbmSimulator(sample_size=1000, hurst_parameter=0.2, lamperti_multiplier=10).get_fbm(is_plot=True,
                                                                                                  seed=1)


def test_first_line_circulant_matrix_neg_fbm():
    dpw_neg_fbm_simulator = DpwNegFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dpw_neg_fbm_simulator._first_line_circulant_matrix(m=20, cov=dpw_neg_fbm_simulator.covariance_line))


def test_dpw_neg_fbm_simulator():
    neg_fbm = DpwNegFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_neg_fbm(is_plot=True, seed=1, y_limits=[-2,2])
    neg_fbm = DpwNegFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_neg_fbm(is_plot=True, seed=1, y_limits=[-2,2])
