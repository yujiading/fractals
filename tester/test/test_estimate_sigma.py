from tester.library.estimate_sigma import EstimateSigma


def test_estimate_sigma():
    cla = EstimateSigma(series=[1,2,3,4], h_series=[0.1,0.2,0.3,0.4])
    print(cla.theta_hat_square)