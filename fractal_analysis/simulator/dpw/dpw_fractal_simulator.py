from typing import Callable
from mpmath import mp, mpf, gamma, sin, hyp2f1

import numpy as np
import scipy

from fractal_analysis.simulator.wood_chan.wood_chan_fractal_simulator import WoodChanFgnSimulator


class DpwSelfSimilarFractalSimulator(WoodChanFgnSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, covariance_func: Callable, 
                 factor: float = None, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            covariance_func: covariance function of the self similar process (elementwise)
            factor: eg /bi/tri factor
            tmax: generates a self similar process using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates a self similar process using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """
        self.dpw_size = sample_size
        scaler = -np.log(1 / self.dpw_size + 1)
        self.fgn_size = np.ceil(scipy.special.lambertw(z=scaler, k=-1).real / scaler).astype(int)
        super().__init__(sample_size=self.fgn_size, hurst_parameter=hurst_parameter, tmax=tmax, std_const=std_const)
        self.covariance_func = covariance_func
        self.factor = factor

    @property
    def _lamperti_subseq_index(self):
        seires_step = self.tmax / self.dpw_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        # shifting negative time index to positive time index
        log_series_t = np.log(series_t) + np.abs(np.log(series_t[0]))
        max_log_series_exp_t = np.max(log_series_t)
        lamperti_subseq_index = np.floor(log_series_t * self.fgn_size / max_log_series_exp_t) - 1
        lamperti_subseq_index[0] = 0
        return lamperti_subseq_index.astype(int)

    def get_self_similar_process(self, is_plot=False, method_name=None, series_name=None, seed=None,
                                 plot_path: str = None, y_limits: list = None):
        seires_step = self.tmax / self.dpw_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        lamp_fgn = self.get_fgn(seed=seed, N=self.fgn_size, cov=self.covariance_line)

        self_similar = series_t ** (self.hurst_parameter * self.factor) * lamp_fgn[self._lamperti_subseq_index]

        if is_plot:
            self.plot(series=self_similar, method_name=method_name, series_name=series_name, save_path=plot_path,
                      y_limits=y_limits)
        return self_similar

    def covariance_with_adaptive_precision(self, k_de, n_de, hurst_de, factor_de):
        """
        Repeatedly calls covariance func with increasing precision until the result stabilizes.
        covariance func should return a Decimal.
        """
        mp.dps = 17
        v = self.covariance_func(k_de=k_de, n_de=n_de, hurst_de=hurst_de, factor_de=factor_de)
        return float(v)

    def covariance_line(self, k):
        n_de = mpf(str(self.fgn_size))
        hurst_de = mpf(str(self.hurst_parameter))
        if self.factor is not None:
            factor_de = mpf(str(self.factor))
        else:
            factor_de = None
        v = np.array([
            self.covariance_with_adaptive_precision(k_de=mpf(str(k_ele)), n_de=n_de, hurst_de=hurst_de,
                                                    factor_de=factor_de) for
            k_ele in k])
        return v


class DpwSubFbmSimulator(DpwSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float,
                 tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            tmax: generates sub-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates sub-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, factor=1,
                         covariance_func=self.sub_fbm_covariance_func, tmax=tmax, std_const=std_const)

    def sub_fbm_covariance_func(self, k_de, n_de, hurst_de, factor_de):
        h2 = 2 * hurst_de
        n = n_de
        k_h_n = k_de * hurst_de / n_de
        k_n_2 = k_de / n_de / 2
        v = n ** k_h_n + n ** (-k_h_n) - (
                (n ** k_n_2 + n ** (-k_n_2)) ** h2 + abs(n ** k_n_2 - n ** (-k_n_2)) ** h2) / 2
        return v

    def get_sub_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        sub_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPW', series_name='Sub-FBM',
                                                plot_path=plot_path, y_limits=y_limits)
        return sub_fbm


class DpwBiFbmSimulator(DpwSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, bi_factor: float, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            bi_factor: (0,1]; when it is 1, the series becomes FBM.
            tmax: generates bi-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates bi-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """
        self.bi_factor = bi_factor
        if self.bi_factor <= 0 or self.bi_factor > 1:
            raise ValueError("bi_factor must be in (0,1]")
        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                         covariance_func=self.bi_fbm_covariance_func, factor=self.bi_factor,
                         tmax=tmax, std_const=std_const)

    def bi_fbm_covariance_func(self, k_de, n_de, hurst_de, factor_de):
        bi_factor_de = factor_de  # for clarity
        k_h_n_de = k_de * hurst_de / n_de
        k_n_2_de = k_de / n_de / 2
        h2 = 2 * hurst_de
        A = n_de ** k_h_n_de
        a = n_de ** (-k_h_n_de)
        B = n_de ** k_n_2_de
        b = n_de ** (-k_n_2_de)
        term1 = (A + a) ** bi_factor_de
        term2 = abs(B - b) ** (h2 * bi_factor_de)
        v = (term1 - term2) / (2 ** bi_factor_de)
        return v

    def get_bi_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        bi_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPW',
                                               series_name=f'{self.bi_factor} Bi-FBM', plot_path=plot_path,
                                               y_limits=y_limits)
        return bi_fbm


class DpwTriFbmSimulator(DpwSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, tri_factor: float, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            tri_factor: (0,1).
            tmax: generates tri-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates tri-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """
        self.tri_factor = tri_factor
        if self.tri_factor <= 0 or self.tri_factor >= 1:
            raise ValueError('tri_factor must be in (0,1)')
        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                         covariance_func=self.tri_fbm_covariance_func, factor=self.tri_factor,
                         tmax=tmax, std_const=std_const)

    def tri_fbm_covariance_func(self, k_de, n_de, hurst_de, factor_de):
        tri_factor = factor_de
        n = n_de
        k_h_f_n = k_de * hurst_de * tri_factor / n_de
        h_k_d = hurst_de * k_de / n_de
        v = n ** k_h_f_n + n ** (-k_h_f_n) - (n ** h_k_d + n ** (-h_k_d)) ** tri_factor
        return v

    def get_tri_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        tri_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPW',
                                                series_name=f'{self.tri_factor} Tri-FBM', plot_path=plot_path,
                                                y_limits=y_limits)
        return tri_fbm


class DpwFbmSimulator(DpwBiFbmSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            tmax: generates FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, bi_factor=1,
                         tmax=tmax, std_const=std_const)

    def get_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPW', series_name='FBM',
                                            plot_path=plot_path, y_limits=y_limits)
        return fbm


class DpwNegFbmSimulator(DpwSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, tmax: float = 1, std_const: float = 1):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            tmax: generates Neg-FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates Neg-FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
        """

        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, factor=1,
                         covariance_func=self.neg_fbm_covariance_func, tmax=tmax, std_const=std_const)

    def neg_fbm_covariance_func(self, k_de, n_de, hurst_de, factor_de):
        n = n_de
        h = hurst_de
        h2 = 2 * h
        k = k_de
        if k_de == mpf('0'):
            v = (gamma(h + 0.5) ** 2) / (gamma(h2 + 1) * sin(h * mp.pi)) - 1 / h2
        else:
            numerator = (gamma(h + 0.5)) ** 2 * (n ** (h * (k / n)) + n ** (h * (-k / n)) - (
                    (abs(1 - n ** (-k / n))) ** h) * (abs(1 - n ** (k / n))) ** h)
            denominator = 2 * gamma(h2 + 1) * sin(h * mp.pi)
            term = -n ** (-abs(k / n) / 2) / (h + 0.5)
            summ = hyp2f1(0.5 - h, 1, h + 3 / 2, n ** (-abs(k / n)))
            v = numerator / denominator + term * summ
        return v

    def get_neg_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        neg_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPW', series_name='Neg-FBM',
                                                plot_path=plot_path, y_limits=y_limits)
        return neg_fbm
