from fractal_analysis.simulator.wood_chan.wood_chan_multi_fractal_simulator import KMeans, WoodChanMbmSimulator
import numpy as np


def test_kmeans():
    # kmeans = KMeans(n_clusters=3, x=[0.1, 0.2, 0.1, 0.02, 0.05, 0.09, 0.6])
    kmeans = KMeans(n_clusters=3, x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(kmeans.cluster_centers)  # 0.0720    0.1500    0.4000
    print(kmeans.labels)  # [1     3     2     1     1     1     3]-1
    print(kmeans.x_max)  # 0.6000
    print(kmeans.x_min)  # 0.0200


def test_new_gamma():
    assert round(WoodChanMbmSimulator.new_gamma(z=0.3), 4) == 2.9916
    assert round(WoodChanMbmSimulator.new_gamma(z=3), 4) == 2
    assert round(WoodChanMbmSimulator.new_gamma(z=30), 4) == -0.1192
    assert round(WoodChanMbmSimulator.new_gamma(z=-30), 4) == 0.0079
    assert round(WoodChanMbmSimulator.new_gamma(z=-0.3), 4) == -4.3269
    assert round(WoodChanMbmSimulator.new_gamma(z=0), 4) == np.inf
    assert round(WoodChanMbmSimulator.new_gamma(z=1), 4) == 1


def test_ii_func():
    assert round(WoodChanMbmSimulator.ii_func(h=0.1), 4) == 11.0725
    assert round(WoodChanMbmSimulator.ii_func(h=0.3), 4) == 4.3460
    assert round(WoodChanMbmSimulator.ii_func(h=0.5), 4) == 3.1416
    assert round(WoodChanMbmSimulator.ii_func(h=0.75), 4) == 3.3422
    assert round(WoodChanMbmSimulator.ii_func(h=0.99), 4) == 50.9357
    assert round(WoodChanMbmSimulator.ii_func(h=0), 4) == np.inf
    assert round(WoodChanMbmSimulator.ii_func(h=1), 4) == np.inf


def test_fbm_cov_matrix():
    v, err = WoodChanMbmSimulator.fbm_cov_matrix(1, 2, 3, 0.4, 0.5, 0.7, 0.6, 100, np.array([0.1, 0.2, 0.3, 0.4]))
    assert all(np.around(v, 4) == [-0.0003, 0.1325, 0.0511, -0.0233, 0.7126, 0.1193])
    assert round(err, 4) == 0.0491


def test_get_mbm():
    sample_size = 500
    t = np.linspace(0, 1, sample_size)
    holder_exponents = 0.9 + 0.01 * np.sin(4 * np.pi * t)
    WoodChanMbmSimulator(sample_size=sample_size, holder_exponents=holder_exponents).get_mbm(is_plot=True, seed=1,
                                                                                             hurst_name='0.5+0.3sin(4pit)')
