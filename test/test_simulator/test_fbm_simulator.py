from fractal_analysis.simulator.fbm_simulator import FbmSimulator


def test_fbm_simulator_lamperti_subseq_index():
    simulator = FbmSimulator(series_len=10, hurst_parameter=0.2, tmax=15)
    print(simulator._lamperti_subseq_index)
