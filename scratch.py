from experiments import test_single_node, test_network, test_i_over_time, test_total_i_over_time, test_i_over_time_wavelets, test_time_of_max_i, test_multiple_policies, beta_ttp, quarantine_v_travel_ban
from vars import VarGetter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # test_single_node(10)
    # test_network()
    # test_i_over_time(1)
    # test_total_i_over_time(1)
    # test_i_over_time_wavelets(1)
    # test_time_of_max_i(1)
    # test_multiple_policies(3)
    # beta_ttp(1)
    quarantine_v_travel_ban(3)
