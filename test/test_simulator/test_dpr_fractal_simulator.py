from fractal_analysis.simulator.dpr.dpr_fractal_simulator import DprSubFbmSimulator, DprFbmSimulator, \
    DprNegFbmSimulator, DprBiFbmSimulator


def test_lamperti_subseq_index():
    simulator = DprSubFbmSimulator(sample_size=1024, hurst_parameter=0.1, tmax=15)
    print(simulator._lamperti_subseq_index)


def test_first_line_circulant_matrix_sub_fbm():
    dpr_sub_fbm_simulator = DprSubFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dpr_sub_fbm_simulator._first_line_circulant_matrix(m=20, cov=dpr_sub_fbm_simulator.sub_fbm_covariance_func))


def test_dpr_sub_fbm_simulator():
    sub_fbm = DprSubFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_sub_fbm(is_plot=True, seed=1)
    sub_fbm = DprSubFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_sub_fbm(is_plot=True, seed=1)



def test_dpr_bi_fbm_simulator():
    bi_fbm = DprBiFbmSimulator(sample_size=1000, hurst_parameter=0.8, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DprBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DprBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=1).get_bi_fbm(is_plot=True, seed=1)


def test_dpr_fbm_simulator():
    fbm = DprFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_fbm(is_plot=True, seed=1)
    fbm = DprFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_fbm(is_plot=True, seed=1)
    fbm = DprFbmSimulator(sample_size=1000, hurst_parameter=0.2, lamperti_multiplier=10).get_fbm(is_plot=True,
                                                                                                        seed=1)


def test_first_line_circulant_matrix_neg_fbm():
    dpr_neg_fbm_simulator = DprNegFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dpr_neg_fbm_simulator._first_line_circulant_matrix(m=20, cov=dpr_neg_fbm_simulator.neg_fbm_covariance_func))


def test_dpr_neg_fbm_simulator():
    neg_fbm = DprNegFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_neg_fbm(is_plot=True, seed=1)
