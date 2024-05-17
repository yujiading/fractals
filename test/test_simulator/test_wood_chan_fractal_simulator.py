from fractal_analysis.simulator.wood_chan.wood_chan_fractal_simulator import WoodChanFbmSimulator


def test_first_line_circulant_matrix_fbm():
    """
    Compare with FracLab: line= lineC( 1000 , 0.3, 1,1,20 )
    line =
          Columns 1 through 10

            0.0158   -0.0038   -0.0008   -0.0004   -0.0003   -0.0002   -0.0002   -0.0001   -0.0001   -0.0001

          Columns 11 through 20

           -0.0001   -0.0001   -0.0001   -0.0001   -0.0002   -0.0002   -0.0003   -0.0004   -0.0008   -0.0038
    """
    wood_chan_fbm_simulator = WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.3)
    print(wood_chan_fbm_simulator._first_line_circulant_matrix(m=20, cov=wood_chan_fbm_simulator.fbm_cov))


def test_simulate_w():
    print(WoodChanFbmSimulator._simulate_w(m=16))


def test_wood_chan_fbm_simulator_get_fbm():
    print(WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_fbm(is_plot=True, seed=1))
    print(WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.8).get_fbm(is_plot=True, seed=1))

    # print(WoodChanFbmSimulator(sample_size=10, hurst_parameter=0.2).get_fbm(is_plot=True, seed=1))
    # [-0.35866472 - 0.75499087 - 0.57936062 - 0.77244189 - 0.21717808  0.89900659
    # 0.45862348  0.62264595  2.44675227  1.00405535]
