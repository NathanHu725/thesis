from network import DiseaseNetwork 
from disease_node import DiseaseNode
from sir_node import SIRNode
from vars import VarGetter
from seirs_node import SEIRSNode
from seirs_de_node import SEIRSDENode

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from scipy import signal, fft

def quarantine_v_travel_ban(trials=10, city_to_analyze='Chicago'):
    v=VarGetter()

    vals = []

    for quarantine_days in range(0, 15):
        vals2 = []
        for threshold in [.05, .075, .1, .125, .15, .2, 1]:
            v.threshold = threshold 
            v.dvars['quarantine_days'] = quarantine_days

            avg_I_data, _ = get_avg_data(trials, v)
            vals2.append(avg_I_data.idxmax(axis=1)[city_to_analyze])

        vals.append(vals2)

    df = pd.DataFrame(vals, index=np.arange(0, 15), columns=[.05, .075, .1, .125, .15, .2, 1])

    sns.heatmap(df)
    plt.xlabel('Quarantine Days')
    plt.xlabel('Travel Ban Threhold (Fraction of Population)')
    plt.title('Quarantine vs Travel Ban')
    plt.show()

    
def beta_ttp(trials=10):
    v = VarGetter()

    betas, ttpfargo, ttpchicago, ttpcolumbus, ttpwichita = [], [], [], [], []

    for i in tqdm(np.linspace(0, 1, 20), 'Betas'):
        v.dvars['beta'] = i
        betas.append(i)

        avg_I_data, _ = get_avg_data(trials, v)

        ttpfargo.append(avg_I_data.idxmax(axis=1)['Fargo'])
        ttpchicago.append(avg_I_data.idxmax(axis=1)['Chicago'])
        ttpcolumbus.append(avg_I_data.idxmax(axis=1)['Columbus'])
        ttpwichita.append(avg_I_data.idxmax(axis=1)['Wichita'])

    plt.scatter(betas, ttpfargo, color='orange', label='Fargo')
    plt.scatter(betas, ttpchicago, color='blue', label='Chicago')
    plt.scatter(betas, ttpcolumbus, color='green', label='Columbus')
    plt.scatter(betas, ttpwichita, color='red', label='Wichita')
    plt.xlabel('Beta Valus')
    plt.ylabel('Log of Time (Days)')
    plt.title('Beta Values vs Time to Peak for Start City')
    plt.legend()
    plt.show()

def test_multiple_policies(trials=10):
    v = VarGetter()
    
    v.dvars['recovery_rate'] = 1/7
    a, b, c, d, x, y = test_time_of_max_i(trials, False, v)
    plt.scatter(x, y, color='lightgreen')
    plt.plot(x, a * pow(x, 3) + b * pow(x, 2) + c * x + d, color='green', label='1/7')

    v.dvars['recovery_rate'] = 1/10
    a, b, c, d, x, y = test_time_of_max_i(trials, False, v)
    plt.scatter(x, y, color='lightblue')
    plt.plot(x, a * pow(x, 3) + b * pow(x, 2) + c * x + d, color='blue', label='1/10')

    v.dvars['recovery_rate'] = 1/21
    a, b, c, d, x, y = test_time_of_max_i(trials, False, v)
    plt.scatter(x, y, color='lightgrey')
    plt.plot(x, a * pow(x, 3) + b * pow(x, 2) + c * x + d, color='grey', label='1/21')

    v.dvars['recovery_rate'] = 1/14
    a, b, c, d, x, y = test_time_of_max_i(trials, False, v)
    plt.scatter(x, y, color='mistyrose')
    plt.plot(x, a * pow(x, 3) + b * pow(x, 2) + c * x + d, color='red', label='1/14')
    plt.legend()
    plt.xlabel(f"Distances from {v.get_start_nodes()[0]} (Miles)")
    plt.ylabel('Time (Days)')
    plt.title(f"Comparing Max I with Different Recovery Rates with {v.threshold} and {v.dvars['quarantine_days']}")
    plt.show()

def get_avg_data(trials, v):
    avg_I_data = pd.DataFrame([])
    for i in tqdm(range(trials), desc="Trials"):
        good = False
        while(not good):
            try:
                net = DiseaseNetwork(v.get_cities(), v.get_distances(), SEIRSNode, v.get_dvars(), v.get_time_vars(), v.get_travel_vars())
                tracker, _, time_tracker, _ = net.simulate()
                good = True
            except:
                print("Invalid Run")

        distance_from_start_order = np.array(v.get_distances()[np.where(np.array(v.get_cities())[:,0] == v.get_start_nodes()[0])[0][0]]).argsort()
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

def test_time_of_max_i(trials=10, show=True, v=VarGetter()):
    avg_I_data, time_tracker = get_avg_data(trials, v)

    distances = pd.DataFrame(np.array(v.get_distances()[np.where(np.array(v.get_cities())[:,0] == v.get_start_nodes()[0])[0][0]]), columns=["Distances"], index=np.array(v.get_cities())[:,0])
    populations = pd.DataFrame(v.get_cities(), columns=['Cities', 'Population']).set_index('Cities').apply(np.log)
    city_vals = pd.concat([distances, populations], axis=1).sort_values(by="Distances")

    colors = ["green" if i == 'Columbus' else "blue" if i == 'Chicago' else "red" if i == 'Fargo' else "orange" if i == 'Wichita' else "lightgrey" for i in distances.index]

    WIDTH = 1
    SIGNAL = signal.morlet2

    a, b, c, d = np.polyfit(city_vals['Distances'], avg_I_data.idxmax(axis=1), 3)
    if(show):
        _, ax = plt.subplots(1, 3)
        ax[0].scatter(city_vals['Distances'], avg_I_data.idxmax(axis=1), color=colors)
        ax[0].plot(city_vals['Distances'], a * pow(city_vals['Distances'], 3) + b * pow(city_vals['Distances'], 2) + c * city_vals['Distances'] + d)
        ax[0].set_title("Max I Value")
        ax[0].set_ylabel("Time in Days")
        ax[0].set_xlabel(f"Distances from {v.get_start_nodes()[0]}")

        ax[1].scatter(city_vals['Population'], avg_I_data.idxmax(axis=1), color=colors)
        ax[1].set_title("Max I Value")
        ax[1].set_ylabel("Time in Days")
        ax[1].set_xlabel(f"Population")

        # max_wavelet = []
        # for i in avg_I_data.index:
        #     max_wavelet.append(np.argmax(abs(signal.cwt(avg_I_data.loc[i], SIGNAL, [1]))) * get_time_vars()['time_step'])

        # ax[1].scatter(distances, max_wavelet, color=colors)
        # ax[1].set_title("Max value of wavelet transform")

        for i in avg_I_data.index:
            color = 'lightgray'
            cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
            ax[2].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color)

        for i in ['Fargo', 'Columbus', 'Chicago', 'Wichita']:
            if i == 'Fargo':
                color = 'r'
                cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
                ax[2].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
            elif i == 'Columbus':
                color = 'g'
                cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
                ax[2].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
            elif i == 'Chicago':
                color = 'b'
                cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
                ax[2].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
            elif i == 'Wichita':
                color = 'orange'
                cwtmatr = signal.cwt(avg_I_data.loc[i], SIGNAL, [1])
                ax[2].plot(time_tracker[:], abs(cwtmatr[WIDTH - 1])[:], color, label=i)
            else:
                pass

        ax[2].set_title("Morlet Wavelet Transform of Width 1 For All Cities")
        ax[2].set_ylabel("Incidence with Morlet")
        ax[2].set_xlabel("Time in Days")
        plt.legend(loc='upper right')
        plt.show()

    return (a, b, c, d, city_vals['Distances'], avg_I_data.idxmax(axis=1))

def test_i_over_time_wavelets(trials=10, v=VarGetter()):
    avg_I_data, time_tracker = get_avg_data(trials, v)
    widths = np.arange(1, 31)
    # ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'cwt']

    WIDTH = 1
    SIGNAL = signal.morlet2

    _, ax = plt.subplots(1, 3)
    for i in avg_I_data.index:
        color = 'lightgray'
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
    txt=f"Percent of population infected over time with y axis arranged by distance from {v.get_start_nodes()[0]}. The time period is {v.get_time_vars()['total_time']} days. The travel model used is {v.get_travel_vars()['connection_type']}."
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
    v = VarGetter()
    avg_I_data = pd.DataFrame([])
    infected_times = defaultdict(lambda: 0)
    for i in tqdm(range(trials), desc="Trials"):
        net = DiseaseNetwork(v.get_cities(), v.get_distances(), SEIRSNode, v.get_dvars(), v.get_time_vars(), v.get_travel_vars())
        tracker, _, time_tracker, _ = net.simulate()

        distance_from_start_order = np.array(v.get_distances()[np.where(np.array(v.get_cities())[:,0] == v.get_start_nodes()[0])[0][0]]).argsort()
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
                if start == 0 and percent_pop > .01:
                    start = time 
                
                if start > 0 and percent_pop < .01:
                    end = time 
                    break 
            
            if end == 0:
                end = 200

            infected_times[city] += end - start 

    infected_times = dict((city, total_times / trials) for city, total_times in infected_times.items())
    avg_I_data = avg_I_data.div(trials)
    _, ax = plt.subplots(1, 3)
    sns.heatmap(avg_I_data, ax=ax[0], xticklabels=int(len(time_tracker)/15))
    ax[0].set_title("Infected Population Over Time")
    txt=f"Percent of population infected over time with y axis arranged by distance from {v.get_start_nodes()[0]}. The time period is {v.get_time_vars()['total_time']} days. The travel model used is {v.get_travel_vars()['connection_type']}."
    for i in avg_I_data.index:
        color = "green" if i == 'Columbus' else "blue" if i == 'Chicago' else "red" if i == 'Fargo' else "orange" if i == 'Wichita' else "lightgrey"
        ax[2].plot(time_tracker, avg_I_data.loc[i], color=color)
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
        plt.legend(['S Population', 'E Population', 'I Population', 'R Population', 'Total Population'], loc='upper right')
    
    plt.show()

def test_single_node(trials = 100):
    v = VarGetter()
    delta_t = v.get_time_vars()['time_step']
    total_t = v.get_time_vars()['total_time']
    de_node = SEIRSDENode(2500000, v.get_dvars(), delta_t = delta_t, name = 'Chicago', start_with_disease=True)
    
    curr_time = 0
    time_tracker = []
    de_tracker = []

    for _ in tqdm(range(int(total_t / delta_t)), desc="Simulation"):
        curr_time += delta_t
        de_node.increment()
        de_tracker.append(de_node.get_state())
        time_tracker.append(curr_time)

    st_tracker = np.array([])
    for _ in tqdm(range(trials), desc="Trial"):
        st_node = SEIRSNode(2500000, v.get_dvars(), delta_t = delta_t, name = 'Chicago', start_with_disease=True)
        temp = []
        for _ in tqdm(range(int(total_t / delta_t)), desc="Simulation"):
            st_node.increment()
            temp.append(st_node.get_state())

        temp = np.array(temp)

        if len(st_tracker) > 0:
            st_tracker += temp 
        else:
            st_tracker = temp
        
    st_tracker = st_tracker / trials
    _, ax = plt.subplots(1, 2)
    st_tracker = np.array(st_tracker)
    ax[0].plot(time_tracker, st_tracker[:,0], 'r', label='S')
    ax[0].plot(time_tracker, st_tracker[:,1], 'y', label='E')
    ax[0].plot(time_tracker, st_tracker[:,2], 'b', label='I')
    ax[0].plot(time_tracker, st_tracker[:,3], 'g', label='R')
    ax[0].plot(time_tracker, st_tracker[:,4], 'g--')
    ax[0].set_xlabel('Time (Days)')
    ax[0].set_ylabel('Populations')
    ax[0].set_title('Population Makeup In Chicago Stochastic')

    de_tracker = np.array(de_tracker)
    ax[1].plot(time_tracker, de_tracker[:,0], 'r', label='S')
    ax[1].plot(time_tracker, de_tracker[:,1], 'y', label='E')
    ax[1].plot(time_tracker, de_tracker[:,2], 'b', label='I')
    ax[1].plot(time_tracker, de_tracker[:,3], 'g', label='R')
    ax[1].plot(time_tracker, de_tracker[:,4], 'g--')
    ax[1].set_xlabel('Time (Days)')
    ax[1].set_ylabel('Populations')
    ax[1].set_title('Population Makeup In Chicago Deterministic')    

    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
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

Distance from Chicago vs max_I, total_I

Think about asymptotic behavior of the model, can we change it (super powerful)

look at dS over dI

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

*Overlay the dots for each of the runs, make different colors for each run
Play with initial conditions (start city)
Change immunity loss and infectious period
quarantine, travel ban (is one more effective)

beta vs avg time to peak (see if it matches a curve)

think about how to quantify quarantine vs travel ban, surface plot of travel ban vs quarantine days
think about how the sin_beta function works, how does first case occurrence affect spread
"""