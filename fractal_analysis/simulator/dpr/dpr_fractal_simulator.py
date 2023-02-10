from typing import Callable

import numpy as np
import scipy
from fractal_analysis.simulator.wood_chan.wood_chan_fractal_simulator import WoodChanFgnSimulator


class DprSelfSimilarProcessSimulator(WoodChanFgnSimulator):
    """
        Source paper: Y. Ding, Q. Peng, G. Ren "Simulation of Self-similar Processes using Lamperti Transformation
                      with An Application to Generate Multifractional Brownian Motion" todo:check source paper name
        Corresponding Matlab code by Guangpeng Ren: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer a self-similar process to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the series.
    """

    def __init__(self, sample_size: int, hurst_parameter: float, covariance_func: Callable,
                 lamperti_multiplier: int = 5,
                 tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            covariance_func: covariance function of the self similar process
            lamperti_multiplier: an integer used for Lamperti transform; bigger value (usually <=10) provides more accuracy; default value is 5
            tmax: generates a self similar process using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates a self similar process using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, tmax=tmax, std_const=std_const)
        self.covariance_func = covariance_func
        if not isinstance(lamperti_multiplier, int) and lamperti_multiplier <= 0:
            raise ValueError(f'lamperti_multiplier must be a positive integer.')
        self.lamperti_multiplier = lamperti_multiplier
        self.lamperti_series_len = self.lamperti_multiplier * self.sample_size

    @property
    def _lamperti_subseq_index(self):
        seires_step = self.tmax / self.sample_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        # shifting negative time index to positive time index
        log_series_t = np.log(series_t) + np.abs(np.log(series_t[0]))
        max_log_series_exp_t = np.max(log_series_t)
        lamperti_subseq_index = np.rint(log_series_t * self.lamperti_series_len / max_log_series_exp_t) - 1
        lamperti_subseq_index[0] = 0
        return series_t, lamperti_subseq_index.astype(int)

    def get_self_similar_process(self, is_plot=False, method_name=None, series_name=None, seed=None):
        series_t, lamperti_subseq_index = self._lamperti_subseq_index
        lamp_fgn = self.get_fgn(seed=seed, N=self.lamperti_series_len, cov=self.covariance_func)
        lamp_fgn = lamp_fgn - lamp_fgn[0]
        self_similar = series_t ** self.hurst_parameter * lamp_fgn[lamperti_subseq_index]
        if is_plot:
            self.plot(series=self_similar, method_name=method_name, series_name=series_name)
        return self_similar


class DprSubFbmSimulator(DprSelfSimilarProcessSimulator):
    """
        Source paper: Y. Ding, Q. Peng, G. Ren "Simulation of Self-similar Processes using Lamperti Transformation
                      with An Application to Generate Multifractional Brownian Motion" todo:check source paper name
        Corresponding Matlab code by Guangpeng Ren: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer sub-FBM (self-similar process) to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the sub-FBM series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the sub-FBM series.
    """

    def __init__(self, sample_size: int, hurst_parameter: float,
                 lamperti_multiplier: int = 5, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            lamperti_multiplier: used for Lamperti transform; bigger value (usually <=10) provides more accuracy; default value is 5
            tmax: generates sub-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates sub-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                         covariance_func=self.sub_fbm_covariance_func,
                         lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)

    def sub_fbm_covariance_func(self, k):
        h2 = 2 * self.hurst_parameter
        k_h_n = k * self.hurst_parameter / self.sample_size
        k_n_2 = k / self.sample_size / 2
        v = np.exp(k_h_n) + np.exp(-k_h_n) - 0.5 * (
                (np.exp(k_n_2) + np.exp(-k_n_2)) ** h2 - np.abs(np.exp(k_n_2) - np.exp(-k_n_2)) ** h2)
        return v

    def get_sub_fbm(self, is_plot=False, seed=None):
        sub_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPR', series_name='Sub-FBM')
        return sub_fbm


class DprBiFbmSimulator(DprSelfSimilarProcessSimulator):
    """
        Source paper: Y. Ding, Q. Peng, G. Ren "Simulation of Self-similar Processes using Lamperti Transformation
                      with An Application to Generate Multifractional Brownian Motion" todo:check source paper name
        Corresponding Matlab code by Guangpeng Ren: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer Bi-FBM (self-similar process) to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the Bi-FBM series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the Bi-FBM series.
    """

    def __init__(self, sample_size: int, hurst_parameter: float, bi_factor: float,
                 lamperti_multiplier: int = 5, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            lamperti_multiplier: used for Lamperti transform; bigger value (usually <=10) provides more accuracy; default value is 5
            bi_factor: (0,1]; when it is 1, the series becomes FBM.
            tmax: generates bi-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates bi-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                         covariance_func=self.bi_fbm_covariance_func,
                         lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)
        self.bi_factor = bi_factor

    def bi_fbm_covariance_func(self, k):
        k_h_n = k * self.hurst_parameter / self.sample_size
        k_n_2 = k / self.sample_size / 2
        h2 = 2 * self.hurst_parameter
        v = ((np.exp(k_h_n) + np.exp(-k_h_n)) ** self.bi_factor - np.abs(np.exp(k_n_2) - np.exp(-k_n_2)) ** (
                h2 * self.bi_factor)) / (2 * self.bi_factor)
        return v

    def get_bi_fbm(self, is_plot=False, seed=None):
        bi_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPR', series_name='Bi-FBM')
        return bi_fbm


class DprFbmSimulator(DprBiFbmSimulator):
    """
        Source paper: Y. Ding, Q. Peng, G. Ren "Simulation of Self-similar Processes using Lamperti Transformation
                      with An Application to Generate Multifractional Brownian Motion" todo:check source paper name
        Corresponding Matlab code by Guangpeng Ren: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer FBM (self-similar process) to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the FBM series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the FBM series.
    """

    def __init__(self, sample_size: int, hurst_parameter: float,
                 lamperti_multiplier: int = 5, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            lamperti_multiplier: used for Lamperti transform; bigger value (usually <=10) provides more accuracy; default value is 5
            tmax: generates FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, bi_factor=1,
                         lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)


class DprNegFbmSimulator(DprSelfSimilarProcessSimulator):
    """
        Source paper: Y. Ding, Q. Peng, G. Ren "Simulation of Self-similar Processes using Lamperti Transformation
                      with An Application to Generate Multifractional Brownian Motion" todo:check source paper name
        Corresponding Matlab code by Guangpeng Ren: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer Neg-FBM (self-similar process) to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the Neg-FBM series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the Neg-FBM series.
    """

    def __init__(self, sample_size: int, hurst_parameter: float,
                 lamperti_multiplier: int = 5, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            lamperti_multiplier: used for Lamperti transform; bigger value (usually <=10) provides more accuracy; default value is 5
            tmax: generates Neg-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates Neg-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                         covariance_func=self.neg_fbm_covariance_func,
                         lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)

    def neg_fbm_covariance_func(self, k):
        h2 = 2 * self.hurst_parameter
        numerator = (scipy.special.gamma(self.hurst_parameter + 0.5)) ** 2 * (
                np.exp(self.hurst_parameter * (k / self.sample_size)) + np.exp(
            self.hurst_parameter * (-k / self.sample_size)) - (
                        (np.abs(1 - np.exp(-k / self.sample_size))) ** self.hurst_parameter) * (
                    np.abs(1 - np.exp(k / self.sample_size))) ** self.hurst_parameter)
        denominator = 2 * scipy.special.gamma(h2 + 1) * np.sin(self.hurst_parameter * np.pi)
        term = -np.exp(-np.abs(k / self.sample_size) / 2) / (self.hurst_parameter + 0.5)
        summ = scipy.special.hyp2f1(0.5 - self.hurst_parameter, 1, self.hurst_parameter + 3 / 2,
                                    np.exp(-np.abs(k / self.sample_size)))
        v = numerator / denominator + term * summ
        v[0] = (scipy.special.gamma(self.hurst_parameter + 0.5) ** 2) / (
                scipy.special.gamma(h2 + 1) * np.sin(self.hurst_parameter * np.pi)) - 1 / h2
        return v

    def get_neg_fbm(self, is_plot=False, seed=None):
        neg_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPR', series_name='Neg-FBM')
        return neg_fbm
