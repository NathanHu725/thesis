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

def get_avg_data(trials):
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

    return avg_I_data.div(trials), time_tracker

def test_time_of_max_i(trials=10):
    avg_I_data, time_tracker = get_avg_data(trials)

    distances = pd.DataFrame(np.array(get_distances()[np.where(np.array(get_cities())[:,0] == get_start_nodes()[0])[0][0]]), columns=["Distances"], index=np.array(get_cities())[:,0]).sort_values(by="Distances")
    colors = ["green" if i == 'Columbus' else "blue" if i == 'Chicago' else "red" if i == 'Fargo' else "orange" if i == 'Wichita' else "lightgrey" for i in distances.index]

    WIDTH = 1
    SIGNAL = signal.morlet2

    _, ax = plt.subplots(1, 2)

    ax[0].scatter(distances, avg_I_data.idxmax(axis=1), color=colors)
    ax[0].set_title("Max I Value")
    ax[0].set_ylabel("Time in Days")
    ax[0].set_xlabel(f"Distances from {get_start_nodes()[0]}")

    # max_wavelet = []
    # for i in avg_I_data.index:
    #     max_wavelet.append(np.argmax(abs(signal.cwt(avg_I_data.loc[i], SIGNAL, [1]))) * get_time_vars()['time_step'])

    # ax[1].scatter(distances, max_wavelet, color=colors)
    # ax[1].set_title("Max value of wavelet transform")

    for i in avg_I_data.index:
        color = 'lightgray'
        cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
        ax[1].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color)

    for i in ['Fargo', 'Columbus', 'Chicago', 'Wichita']:
        if i == 'Fargo':
            color = 'r'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
            ax[1].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
        elif i == 'Columbus':
            color = 'g'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
            ax[1].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
        elif i == 'Chicago':
            color = 'b'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
            ax[1].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
        elif i == 'Wichita':
            color = 'orange'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
            ax[1].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
        else:
            pass

    ax[1].set_title("Morlet Wavelet Transform of Width 1 For All Cities")
    ax[1].set_ylabel("Incidence with Morlet")
    ax[1].set_xlabel("Time in Days")
    plt.legend(loc='upper right')
    plt.show()


def test_i_over_time_wavelets(trials=10):
    avg_I_data, time_tracker = get_avg_data(trials)
    widths = np.arange(1, 31)
    # ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt']

    WIDTH = 1
    SIGNAL = signal.morlet2

    _, ax = plt.subplots(1, 3)
    for i in avg_I_data.index:
        color = 'gray'
        cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, widths)
        ax[2].plot(time_tracker, abs(cwtmatr[WIDTH - 1]), color)

    for i in avg_I_data.index:
        if i == 'Fargo':
            color = 'r'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, widths)
            ax[2].plot(time_tracker, abs(cwtmatr[WIDTH - 1]), color)
        elif i == 'Columbus':
            color = 'g'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, widths)
            ax[2].plot(time_tracker, abs(cwtmatr[WIDTH - 1]), color)
        elif i == 'Chicago':
            color = 'b'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, widths)
            ax[2].plot(time_tracker, abs(cwtmatr[WIDTH - 1]), color)
        else:
            pass

    # cwtmatr = signal.cwt(avg_I_data.iloc[0], SIGNAL, widths)
    # for width in cwtmatr[1:]:
    #     ax[1].plot(time_tracker, width)

    ax[1].plot(SIGNAL(M=100, s=WIDTH))

    # ax[1].plot(time_tracker, cwtmatr[-1])
    sns.heatmap(np.flipud(abs(cwtmatr)), ax=ax[0], xticklabels=int(len(time_tracker)/15))
    # sns.heatmap(fft.dct(avg_I_data.values)[:,:20], ax=ax[2], xticklabels=int(len(time_tracker)/15))
    # sns.heatmap(avg_I_data, ax=ax[1], xticklabels=int(len(time_tracker)/15))
    ax[2].set_title(f"All wavelet transforms of width {WIDTH}")
    # ax[1].set_title(f"Infected Population Over Time For {avg_I_data.index[0]}")
    # ax[2].set_title(f"Visualization of Morlet2 for width {WIDTH}")
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
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'], loc='upper right')
    
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
    for i in avg_I_data.index:
        color = "green" if i == 'Columbus' else "blue" if i == 'Chicago' else "red" if i == 'Fargo' else "orange" if i == 'Wichita' else "lightgrey"
        ax[1].plot(time_tracker, avg_I_data.loc[i], color=color)
    # plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    # sns.barplot(x=list(infected_times.keys()), y=list(infected_times.values()), ax=ax[1])
    # plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize='x-small')
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
        plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'], loc='upper right')
    
    plt.show()

def test_single_node():
    net = DiseaseNetwork(get_cities(), get_distances(), SEIRSNode, get_dvars(), get_time_vars(), get_travel_vars())
    tracker, _, time_tracker, peak_I_tracker = net.simulate()
    populations = np.array(tracker['Chicago'])
    S = plt.plot(time_tracker, populations[:,0], 'r')
    E = plt.plot(time_tracker, populations[:,1], 'y')
    I = plt.plot(time_tracker, populations[:,2], 'b')
    R = plt.plot(time_tracker, populations[:,3], 'g')
    cS = plt.plot(time_tracker, populations[:,4], 'g--')
    plt.xlabel('Time')
    plt.ylabel('Disease Populations')
    plt.title('Chicago')    
    plt.legend(['S Population', 'E Population', 'I Population', 'R Population'], loc='upper right')
    plt.show()

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

look at dS over dI


Use a discrete time model of the ODE to show correctness when compared to the stochastic model, do for one city also talk about keeping populations constant

Play around to see rough observations, start making some observations, how can we compare things
- Three indicator cities
- Travel ban cities
- What else can we change in the model
    - Types of travel
    - Quarantine
    - Thresholding
    - Immunity loss
    - Testing Variables
- Line of best fit with confidence interval (logarithmic or polynomial (order 2 or 3))

Plot against population size

*Overlay the dots for each of the runs, make different colors for each run
Play with initial conditions (start city)
Change immunity loss and infectious period
quarantine, travel ban (is one more effective)

"""