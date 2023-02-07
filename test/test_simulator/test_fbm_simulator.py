from fractal_analysis.simulator.fbm_simulator import FbmSimulator, WoodChanFbmSimulator


def test_first_line_circulant_matrix():
    """
    Compare with FracLab: line= lineC( 1000 , 0.3, 1,1,20 )
    line =
          Columns 1 through 10

            0.0158   -0.0038   -0.0008   -0.0004   -0.0003   -0.0002   -0.0002   -0.0001   -0.0001   -0.0001

          Columns 11 through 20

           -0.0001   -0.0001   -0.0001   -0.0001   -0.0002   -0.0002   -0.0003   -0.0004   -0.0008   -0.0038
    """
    print(WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.3)._first_line_circulant_matrix(m=20))


def test_simulate_w():
    print(WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.3)._simulate_w(m=16))

def test_woodchanfbmsimulator_fbm():
    print(WoodChanFbmSimulator(sample_size=1000, hurst_parameter=0.2).get_fbm(is_plot=True))

def test_fbm_simulator_lamperti_subseq_index():
    simulator = FbmSimulator(series_len=10, hurst_parameter=0.2, tmax=15)
    print(simulator._lamperti_subseq_index)
