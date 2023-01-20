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

    def logplusone(a):
        return np.log10(a+1)

    data = pd.DataFrame(I_populations, columns=time_tracker, index=city_list).apply(logplusone)

    _, ax = plt.subplots()
    ax = sns.heatmap(data, ax=ax, xticklabels=int(len(time_tracker)/15))
    plt.title("Infected Population Over Time")

    txt=f"Percent of population infected over time with y axis arranged by distance from {get_start_nodes()[0]}. The time period is {get_time_vars()['total_time']} days. The travel model used is {get_travel_vars()['connection_type']}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.show()

def test_network(): 
    net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
    tracker, _, time_tracker, peak_I_tracker = net.simulate()
    cities = ['Chicago', 'Milwaukee', 'Rockford', 'Gary', 'St. Louis', 'Columbus', 'Sioux Falls', 'Fargo']
    for city, number in zip(cities, range(1, len(cities) + 1)):
        populations = np.array(tracker[city])
        plt.subplot(2,4,number)
        S = plt.plot(time_tracker, populations[:,0], 'r')
        E = plt.plot(time_tracker, populations[:,1], 'y')
        I = plt.plot(time_tracker, populations[:,2], 'b')
        R = plt.plot(time_tracker, populations[:,3], 'g')
        cS = plt.plot(time_tracker, populations[:,4], 'g--')
        plt.xlabel('Time')
        plt.ylabel('Disease Populations')
        plt.title(city)
        plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    
    plt.show()

def test_single_node():
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
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population'])
    plt.show()

if __name__ == "__main__":
    # test_two_nodes()
    # test_single_node()
    # test_network()
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

Work on implementing radiation
Work on making the travel ban go both directions

Think about asymptotic behavior of the model, can we change it (super powerful)
Heatmap of time vs city where each row is the infections over time (try 1 year to start with, can also look at sqrt(I_j(t) or log(I_j(t) + 1)))

For quarantine period, think about making sure not all people come out recovered (binomial distribution to the power of the number of days people are in quarantine)

Same set of parameters, run it once for each city starting. Compare how city size and distance from chicago or distance from center of network (think about summing all 
distances, connectedness to network)

Run averages over a number of trials
"""