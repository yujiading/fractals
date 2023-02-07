from typing import Union, List, Annotated

import numpy as np
import pandas as pd
import math
import random
from scipy.fftpack import fft
import matplotlib.pyplot as plt


class WoodChanFbmSimulator:
    """
        Main idea: WoodChanFbmSimulator generates a Fractional Brownian Motion (fBm) using Wood and Chan circulant matrix.
        Source paper: A.T.A. Wood, G. Chan, "Simulation of stationary Gaussian process in [0,1]d",
%                     Journal of Computational and Graphical Statistics, Vol. 3 (1994) 409-432.
        Corresponding Matlab code: FracLab (https://project.inria.fr/fraclab/)
                                   fbmwoodchan.m
    """

    def __init__(self, sample_size: int,
                 hurst_parameter: float,
                 tmax: float = 1,
                 std_const: float = 1,
                 seed: int = None):
        """
            sample_size: is the length of samples generated; it is a positive integer.
            hurst_parameter: is a real value in (0:1) that governs both the pointwise regularity and the shape around 0 of the power spectrum.
            tmax: generates FBM using a specific size of time support, i.e. the time runs in [0,tmax]; if tmax is not specified, the default value is tmax = 1.
            std_const: generates FBM using a specific standard deviation at instant t = 1; if std_const is not specified, the default value is std_const = 1.
            seed: generates FBM with a specific random seed.
        """
        self.sample_size = sample_size
        if self.sample_size <= 0:
            raise ValueError(f'sample_size must be a positive integer.')
        self.hurst_parameter = hurst_parameter
        if not (0 < self.hurst_parameter < 1):
            raise ValueError(f'hurst_parameter must be in range (0, 1).')
        self.tmax = tmax
        if not self.tmax > 0:
            raise ValueError(f'tmax must be positive.')
        self.std_const = std_const
        if not self.std_const > 0:
            raise ValueError(f'std_const must be positive.')
        self.seed = seed

    def _first_line_circulant_matrix(self, m):
        """
            First line of the circulant matrix C built with covariances of the stationary process.
            References : 1 - Wood and Chan (1994) 2 - Phd Thesis Coeurjolly (2000), Appendix A p.132
            m : power of two larger than 2*(n-1).
        """
        k = self.tmax * np.arange(0, m)
        h2 = 2 * self.hurst_parameter
        v = 0.5 * self.std_const ** 2 * (abs(k - self.tmax) ** h2 - 2 * k ** h2 + (k + self.tmax) ** h2)
        ind = np.concatenate((np.arange(0, m / 2, dtype=int), np.arange(m / 2, 0, -1, dtype=int)))
        line = v[ind] / self.sample_size ** h2
        return line

    def _simulate_w(self, m):
        # simulation of W=(Q)^t Z, where Z leads N(0,I_m) and (Q)_{jk} = m^(-1/2) exp(-2i pi jk/m)
        np.random.seed(self.seed)
        ar = np.random.normal(0, 1, int(m / 2 + 1))
        ai = np.random.normal(0, 1, int(m / 2 + 1))
        ar[0] = 2 ** 0.5 * ar[0]
        ar[-1] = 2 ** 0.5 * ar[-1]
        ai[0] = 0
        ai[-1] = 0
        ar = np.concatenate((ar, ar[int(m / 2 - 1): 0:-1]))
        aic = -ai
        ai = np.concatenate((ai, aic[int(m / 2 - 1): 0:-1]))
        W = [complex(one_ar, one_ai) for one_ar, one_ai in zip(ar, ai)]
        return W

    def get_fbm(self, is_plot=False):
        # Construction of the first line of the circulant matrix C
        m = 2 ** (int(math.log(self.sample_size - 1) / math.log(2) + 1))
        eigC = self._first_line_circulant_matrix(m)
        eigC = fft(eigC).real
        # search of the power of two (<2**18) such that eigC is definite positive
        while any(v <= 0 for v in eigC) and m < 2 ** 17:
            m = 2 * m
            eigC = self._first_line_circulant_matrix(m)
            eigC = fft(eigC).real
        # simulation of W=(Q)^t Z, where Z leads N(0,I_m) and (Q)_{jk} = m^(-1/2) exp(-2i pi jk/m)
        W = self._simulate_w(m=m)
        # reconstruction of the fGn
        W = np.sqrt(eigC) * W
        fGn = fft(W)
        fGn = fGn / (2 * m) ** 0.5
        fGn = fGn.real
        fGn = fGn[:self.sample_size]
        fBm = np.cumsum(fGn)
        if is_plot:
            plt.plot(np.arange(0, self.tmax,self.tmax/self.sample_size),fBm)
            plt.title(f'Wood Chan FBM simulation with {self.sample_size} samples and {self.hurst_parameter} hurst')
            plt.xlabel('Time')
            plt.show()
        return fBm



class FbmSimulator:
    """
        Source paper: todo:add source paper
        Corresponding Matlab code: todo:add link to matlab code
        Main idea: we use Lamperti transform to transfer FBM (self-similar process) to a stationary process, and
                   simulate the stationary process using circulant embedding approach (Wood, A.T.A., Chan, G., 1994.
                   Simulation of stationary Gaussian processes in [0, 1]^d. Journal of computational and graphical
                   statistics 3, 409–432). Then a subsequence of the simulated stationary process is convert back to
                   the FBM series. Because we need to use its subsequence, the stationary process needs to have a longer
                   length (lamperti_series_len_multiplier*series_len) than the FBM series.
    """

    def __init__(self,
                 series_len: int,
                 hurst_parameter: float,
                 lamperti_series_len_multiplier: int = 5,
                 tmax: float = 1,
                 const: float = 1,
                 seed: int = None):
        """
            series_len: length of the fbm series to simulate
            hurst_index: hurst parameter of fbm
            lamperti_series_len_multiplier: bigger value (usually <=10) provides more accuracy; default value is 5
            tmax: the time runs in [0,tmax]; the default value is 1
            const: standard derivation at initial point; default value is 1

        """
        self.series_len = series_len
        self.hurst_parameter = hurst_parameter
        if not (0 < self.hurst_parameter < 1):
            raise ValueError(f'Hurst index must be in range (0, 1).')
        self.tmax = tmax
        if not self.tmax > 0:
            raise ValueError(f'tmax must be positive.')
        self.const = const
        if not self.const > 0:
            raise ValueError(f'const must be positive.')
        self.seed = seed
        self.lamperti_series_len_multiplier = lamperti_series_len_multiplier
        self.lamperti_series_len = self.lamperti_series_len_multiplier * self.series_len

    @property
    def _lamperti_subseq_index(self):
        seires_step = self.tmax / self.series_len
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        # transfer time to (1,e)
        series_exp_t = (np.exp(1) - 1) / self.tmax * series_t + 1
        # log of time in (1,e) is in (0,1)
        log_series_exp_t = np.log(series_exp_t)
        max_log_series_exp_t = np.max(log_series_exp_t)
        lamperti_subseq_index = np.rint(log_series_exp_t * self.lamperti_series_len / max_log_series_exp_t) - 1
        return lamperti_subseq_index.astype(int)

    def _line_circulant(self, m):
        """
            First line of the circulant matrix C built with covariances of the stationary process.
        """
