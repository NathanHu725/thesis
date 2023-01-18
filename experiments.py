from network import DiseaseNetwork 
from disease_node import DiseaseNode
from sir_node import SIRNode
from vars import get_cities, get_distances, get_dvars, get_time_vars, get_travel_vars, get_start_nodes
from seirs_node import SEIRSNode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def test_i_over_time():
    net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
    tracker, _, time_tracker, _ = net.simulate()

    distance_from_start_order = np.array(get_distances()[np.where(np.array(get_cities())[:,0] == get_start_nodes()[0])[0][0]]).argsort()
    city_list = np.array(list(tracker))[distance_from_start_order]
    I_populations = np.array([[i[2] / i[4] for i in np.array(city_stats)] for city_stats in tracker.values()])[distance_from_start_order]

    data = pd.DataFrame(I_populations, columns=time_tracker, index=city_list)

    _, ax = plt.subplots()
    ax = sns.heatmap(data, ax=ax, xticklabels=int(len(time_tracker)/15))
    plt.title("Infected Population Over Time")

    txt=f"Percent of population infected over time with y axis arranged by distance from {get_start_nodes()[0]}. The time period is {get_time_vars()['total_time']} days. The travel model used is {get_travel_vars()['connection_type']}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.show()

def test_network(): 
    net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
    tracker, _, time_tracker, peak_I_tracker = net.simulate()
    populations = np.array(tracker['Chicago'])
    plt.subplot(2,4,1)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Chicago')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    plt.subplot(2,4,2)
    populations = np.array(tracker['Milwaukee'])
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Milwaukee')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['Rockford'])
    plt.subplot(2,4,3)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Rockford')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['Gary'])
    plt.subplot(2,4,4)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Gary')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['St. Louis'])
    plt.subplot(2,4,5)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('St. Louis')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['Columbus'])
    plt.subplot(2,4,6)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Columbus')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['Sioux Falls'])
    plt.subplot(2,4,7)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Sioux Falls')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    populations = np.array(tracker['Fargo'])
    plt.subplot(2,4,8)
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Fargo')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    plt.show()

def test_single_node():
    N = 100000
    time = 250

    dvars = {'beta': 1.5/5, 
            'birth_rate': 0,
            'natural_death_rate': 0,
            'disease_death_rate': 0,
            'incubation_rate': 1/3,
            'recovery_rate': 1/5,
            'lost_immunity_rate': 1/50} 
            # Test one year, five years, twenty five

    S_values, E_values, I_values, R_values, time_steps = [], [], [], [], []

    test_node = SEIRSNode(N, disease_vars=dvars, delta_t = 1)

    for i in range(time):
        print(test_node.get_state())
        new_S, new_E, new_I, new_R = test_node.get_state()
        S_values.append(new_S)
        E_values.append(new_E)
        I_values.append(new_I)
        R_values.append(new_R)
        time_steps.append(i)
        test_node.increment()
        
    S = plt.plot(time_steps, S_values, 'r')
    E = plt.plot(time_steps, E_values, 'y')
    I = plt.plot(time_steps, I_values, 'b')
    R = plt.plot(time_steps, R_values, 'g')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population'])
    plt.show()

def test_two_nodes():
    N1 = 500000
    time = 600

    S1_values, E1_values, I1_values, R1_values, time_steps = [], [], [], [], []

    dvars = {'beta': 1.5/5, 
            'birth_rate': 1 / (65 * 365),
            'natural_death_rate': 1 / (75 * 365),
            'disease_death_rate': .01,
            'incubation_rate': 1/3,
            'recovery_rate': 1/5,
            'lost_immunity_rate': 1/50} 
    

    test_node_1 = SEIRSNode(N1, disease_vars=dvars, delta_t = 1)

    N2 = 200000

    S2_values, E2_values, I2_values, R2_values = [], [], [], []

    test_node_2 = SEIRSNode(N2, disease_vars=dvars, delta_t = .1)

    test_node_1.add_travel_information(test_node_2, .05)
    test_node_2.add_travel_information(test_node_1, .02)

    for i in range(time):
        # print(test_node.get_state())
        new_S1, new_E1, new_I1, new_R1 = test_node_1.get_state()
        S1_values.append(new_S1)
        E1_values.append(new_E1)
        I1_values.append(new_I1)
        R1_values.append(new_R1)
        new_S2, new_E2, new_I2, new_R2 = test_node_2.get_state()
        S2_values.append(new_S2)
        E2_values.append(new_E2)
        I2_values.append(new_I2)
        R2_values.append(new_R2)
        time_steps.append(i * 1)
        test_node_1.increment()
        test_node_2.increment()
        test_node_1.apply_travel()
        test_node_2.apply_travel()

    plt.subplot(1,2,1)
    S = plt.plot(time_steps, S1_values, 'r')
    E = plt.plot(time_steps, E1_values, 'y')
    I = plt.plot(time_steps, I1_values, 'b')
    R = plt.plot(time_steps, R1_values, 'g')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Town 1')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population'])
    
    plt.subplot(1,2,2)
    S = plt.plot(time_steps, S2_values, 'r')
    E = plt.plot(time_steps, E2_values, 'y')
    I = plt.plot(time_steps, I2_values, 'b')
    R = plt.plot(time_steps, R2_values, 'g')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Town 2')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population'])
    
    plt.show()

if __name__ == "__main__":
    # test_two_nodes()
    test_network()
    test_i_over_time()

"""
Think about questions to ask
Read two travel papers
Think about how to graph data so that my question is answered (peak(I), cumulative(I))
Start tracking cumulative I (total number of people infected, new ones at each step)
parameter that is varying, propbability of extinction (of disease)
cumulative death

Time to spikes or inbetween spikes
Think about the ocsillations
Max oscillation, min oscillation, time inbetween

Population size vs max_I, total_I
Distance from Chicago vs max_I, total_I

Work on making travel make sense (why is naperville seeing no one)
Work on implementing radiation

Think about asymptotic behavior of the model, can we change it (super powerful)
Heatmap of time vs city where each row is the infections over time (try 1 year to start with, can also look at sqrt(I_j(t) or log(I_j(t) + 1)))

For quarantine period, think about making sure not all people come out recovered (binomial distribution to the power of the number of days people are in quarantine)
"""