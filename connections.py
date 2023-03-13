from abc import ABC, abstractmethod
import numpy as np 

class Connection(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_travel(self, threshold, curr_pop):
        pass

class GravityConnection(Connection):
    def __init__(self, name, nodes, destination_node, travel_vars, distance):
        self.node = destination_node 
        self.A = travel_vars['A']
        self.gamma = travel_vars['gamma']
        self.distance = distance

    def get_travel(self, threshold, infected, curr_pop):
        if threshold(infected, curr_pop):
            return 0

        return self.A * pow(abs(curr_pop), .7) * pow(abs(self.node.get_population()), .7) / pow(self.distance, self.gamma)

class RadiationConnection(Connection):
    def __init__(self, name, nodes, destination_node, travel_vars, distance):
        self.node = destination_node
        self.included_nodes = []
        self.name = name
        for node, node_distance in zip(nodes.values(), travel_vars['adjacency_matrix'][list(nodes.keys()).index(name)]):
            if node_distance < distance and node_distance > 0:
                self.included_nodes.append(node)

        self.T_i = travel_vars['commuter_proportion'] #* ((np.log(3000000 / nodes[name].get_population())) + 1) #* 3 * (len(self.included_nodes)) / len(nodes.values())
        # * 3 * len(self.included_nodes) / len(nodes.values())
        # ((np.log(2500000 / nodes[name].get_population()) / np.log(3)) + .2) *

    def get_travel(self, threshold, infected, curr_pop):
        if threshold(infected, curr_pop):
            return 0

        s_ij = sum([node.get_population() for node in self.included_nodes if node != self.node])
        test = curr_pop * self.T_i * (curr_pop * self.node.get_population()) / ((curr_pop + s_ij) * (curr_pop + self.node.get_population() + s_ij))
        return test