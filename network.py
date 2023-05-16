from disease_node import DiseaseNode
from sir_node import SIRNode
from seirs_node import SEIRSNode
from vars import VarGetter

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class DiseaseNetwork:
    def __init__(self, node_info, adjacency_matrix, node_type, disease_vars, time_vars, travel_vars):
        self.node_type = node_type
        self.disease_vars = disease_vars
        self.time_step = time_vars['time_step'] 
        self.total_time = time_vars['total_time']
        self.travel_vars = travel_vars
        self.graph, self.node_names, self.population_tracker, self.peak_I_tracker = self.create_graph(node_info, adjacency_matrix)

    def create_graph(self, node_info, adjacency_matrix):
        graph = {}
        node_names = []
        population_tracker = {}
        peak_I_tracker = {}
        for node_name, population in node_info:
            new_node = self.node_type(total_population = population, disease_vars = self.disease_vars, delta_t = self.time_step, name=node_name, start_with_disease=node_name in VarGetter().get_start_nodes())
            graph[node_name] = new_node
            node_names.append(node_name)
            population_tracker[node_name] = []
            peak_I_tracker[node_name] = []
            
        nodes_iter = iter(node_names)
        for connections in adjacency_matrix:
            curr_node = next(nodes_iter)
            for dist, destination_node in zip(connections, node_names):
                if destination_node != curr_node:
                    graph[curr_node].add_travel_information(nodes=graph, target_node=graph[destination_node], travel_vars=self.travel_vars, dist=dist)

        return graph, node_names, population_tracker, peak_I_tracker

    def simulate(self):
        curr_time = 0
        time_tracker = []
        for _ in range(int(self.total_time / self.time_step)):
            curr_time += self.time_step
            self.increment()
            time_tracker.append(curr_time)

        for name, node in self.graph.items():
            self.peak_I_tracker[name] = node.peak_I

        return self.population_tracker, self.graph, time_tracker, self.peak_I_tracker
    
    def increment(self):
        # Increment node populations
        for node in self.graph.values():
            node.increment()

        # This needs to be done after all incrementing
        for node in self.graph.values():
            node.apply_travel()
            self.population_tracker[node.name].append(node.get_state())


    def get_population_tracker(self):
        return self.population_tracker        

    # Returns the state of each node in a list
    # Also returns a string of what each place in the state represents
    def get_state(self):
        states = []
        header = None
        for node in self.graph:
            states.append(node.get_state())
            if not header:
                header = node.get_header()

        return states, header

    def get_names(self):
        return self.node_names