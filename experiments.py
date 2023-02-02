from network import DiseaseNetwork 
from disease_node import DiseaseNode
from sir_node import SIRNode
from vars import get_cities, get_distances, get_dvars, get_time_vars, get_travel_vars, get_start_nodes
from seirs_node import SEIRSNode

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from scipy import signal, fft

def test_i_over_time_wavelets(trials=10):
    avg_I_data = pd.DataFrame([])
    for i in tqdm(range(trials), desc="Trials"):
        try:
            net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
        except:
            print("Invalid Run")
        tracker, _, time_tracker, _ = net.simulate()

        distance_from_start_order = np.array(get_distances()[np.where(np.array(get_cities())[:,0] == get_start_nodes()[0])[0][0]]).argsort()
        city_list = np.array(list(tracker))[distance_from_start_order]
        I_populations = np.array([[i[2] / i[4] for i in np.array(city_stats)] for city_stats in tracker.values()])[distance_from_start_order]

        def logplusone(a):
            return np.log10(a+1)
        data = pd.DataFrame(I_populations, columns=time_tracker, index=city_list).apply(logplusone)

        if avg_I_data.empty:
            avg_I_data = data 
        else:
            avg_I_data += data 

    avg_I_data = avg_I_data.div(trials)

    widths = np.arange(1, 31)
    # ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt']
    cwtmatr = signal.cwt(avg_I_data.iloc[0], signal.morlet2, widths)

    _, ax = plt.subplots(1, 3)
    ax[0].plot(time_tracker, cwtmatr[0])
    ax[1].plot(time_tracker, cwtmatr[-1])
    # sns.heatmap(np.flipud(cwtmatr), ax=ax[0], xticklabels=int(len(time_tracker)/15))
    # sns.heatmap(fft.dct(avg_I_data), ax=ax[0], xticklabels=int(len(time_tracker)/15))
    sns.heatmap(avg_I_data, ax=ax[2], xticklabels=int(len(time_tracker)/15))
    ax[0].set_title(f"Infected Population Over Time For {avg_I_data.index[0]}")
    txt=f"Percent of population infected over time with y axis arranged by distance from {get_start_nodes()[0]}. The time period is {get_time_vars()['total_time']} days. The travel model used is {get_travel_vars()['connection_type']}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

def test_total_i_over_time(trials=10):
    avg_populations = pd.DataFrame([])
    for i in tqdm(range(trials), desc="Trials"):
        net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
        tracker, _, time_tracker, _ = net.simulate()
        populations = pd.DataFrame(tracker[list(tracker)[0]])
        for city in list(tracker)[1:]:
            populations += pd.DataFrame(tracker[city])

        if avg_populations.empty:
            avg_populations = populations 
        else:
            avg_populations += populations

    S = plt.plot(time_tracker, avg_populations.iloc[:,0], 'r')
    E = plt.plot(time_tracker, avg_populations.iloc[:,1], 'y')
    I = plt.plot(time_tracker, avg_populations.iloc[:,2], 'b')
    R = plt.plot(time_tracker, avg_populations.iloc[:,3], 'g')
    cS = plt.plot(time_tracker, avg_populations.iloc[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Total Populations over Time Starting in Joliet')
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'])
    
    plt.show()

def test_i_over_time(trials=10):
    avg_I_data = pd.DataFrame([])
    infected_times = defaultdict(lambda: 0)
    for i in tqdm(range(trials), desc="Trials"):
        net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
        tracker, _, time_tracker, _ = net.simulate()

        distance_from_start_order = np.array(get_distances()[np.where(np.array(get_cities())[:,0] == get_start_nodes()[0])[0][0]]).argsort()
        city_list = np.array(list(tracker))[distance_from_start_order]
        I_populations = np.array([[i[2] / i[4] for i in np.array(city_stats)] for city_stats in tracker.values()])[distance_from_start_order]

        def logplusone(a):
            return np.log10(a+1)
        # .apply(logplusone)
        data = pd.DataFrame(I_populations, columns=time_tracker, index=city_list)

        if avg_I_data.empty:
            avg_I_data = data 
        else:
            avg_I_data += data 

        for city in data.index:
            start = 0
            end = 0
            for time, percent_pop in data.loc[city].iteritems():
                if start == 0 and percent_pop > .02:
                    start = time 
                
                if start > 0 and percent_pop < .02:
                    end = time 
                    break 
            
            if end == 0:
                end = 200

            infected_times[city] += end - start 

    infected_times = dict((city, total_times / trials) for city, total_times in infected_times.items())
    avg_I_data = avg_I_data.div(trials)
    figures, ax = plt.subplots(1, 2)
    sns.heatmap(avg_I_data, ax=ax[0], xticklabels=int(len(time_tracker)/15))
    ax[0].set_title("Infected Population Over Time")
    txt=f"Percent of population infected over time with y axis arranged by distance from {get_start_nodes()[0]}. The time period is {get_time_vars()['total_time']} days. The travel model used is {get_travel_vars()['connection_type']}."
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    sns.barplot(x=list(infected_times.keys()), y=list(infected_times.values()), ax=ax[1])
    plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
    plt.show()

def test_network(): 
    net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
    tracker, _, time_tracker, peak_I_tracker = net.simulate()
    cities = ['Chicago', 'Milwaukee', 'Rockford', 'Gary', 'St. Louis', 'Columbus', 'Independence', 'Olathe']
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
    # test_i_over_time(1)
    # test_total_i_over_time()
    test_i_over_time_wavelets(1)

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

Think about asymptotic behavior of the model, can we change it (super powerful)

Same set of parameters, run it once for each city starting. Compare how city size and distance from chicago or distance from center of network (think about summing all 
distances, connectedness to network)

Cities that avoid an epidemic see starting in a rural area vs a city, have number of quarantine days on the x axis, also include testing rate

could look at cumulative cases within a node to do the thresholding, average total population for each node

look at rohab (the one with the map), look at a single componenet of the wavelet
understand what width means, how we can choose
how to compute residual phase angles (what does this mean) and graph this vs distance from chicago
maybe look at fourier transforms?

look at dS over dI
"""