from fractal_analysis.simulator.dprw.dprw_fractal_simulator import DprwSubFbmSimulator, DprwFbmSimulator, \
    DprwNegFbmSimulator, DprwBiFbmSimulator


def test_lamperti_subseq_index():
    simulator = DprwSubFbmSimulator(sample_size=1024, hurst_parameter=0.1, tmax=15)
    print(simulator._lamperti_subseq_index)


def test_first_line_circulant_matrix_sub_fbm():
    dprw_sub_fbm_simulator = DprwSubFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dprw_sub_fbm_simulator._first_line_circulant_matrix(m=20, cov=dprw_sub_fbm_simulator.sub_fbm_covariance_func))


def test_dprw_sub_fbm_simulator():
    sub_fbm = DprwSubFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_sub_fbm(is_plot=True, seed=1)
    sub_fbm = DprwSubFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_sub_fbm(is_plot=True, seed=1)



def test_dprw_bi_fbm_simulator():
    bi_fbm = DprwBiFbmSimulator(sample_size=1000, hurst_parameter=0.8, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DprwBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=0.2).get_bi_fbm(is_plot=True, seed=1)
    bi_fbm = DprwBiFbmSimulator(sample_size=1000, hurst_parameter=0.2, bi_factor=1).get_bi_fbm(is_plot=True, seed=1)


def test_dprw_fbm_simulator():
    fbm = DprwFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_fbm(is_plot=True, seed=1)
    fbm = DprwFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_fbm(is_plot=True, seed=1)
    fbm = DprwFbmSimulator(sample_size=1000, hurst_parameter=0.2, lamperti_multiplier=10).get_fbm(is_plot=True,
                                                                                                  seed=1)


def test_first_line_circulant_matrix_neg_fbm():
    dprw_neg_fbm_simulator = DprwNegFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(dprw_neg_fbm_simulator._first_line_circulant_matrix(m=20, cov=dprw_neg_fbm_simulator.neg_fbm_covariance_func))


def test_dprw_neg_fbm_simulator():
    neg_fbm = DprwNegFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_neg_fbm(is_plot=True, seed=1)
