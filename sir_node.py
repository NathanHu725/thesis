from disease_node import DiseaseNode
import numpy as np

class SIRNode(DiseaseNode):
    def __init__(self, total_population, disease_vars, delta_t = 1, name = ''):
        self.N = total_population 
        self.t = 0
        self.delta_t = delta_t
        self.S = self.N - 100
        self.I = 100
        self.R = 0

        self.total_I = 100
        self.peak_I = 0

        self.name = name

        try:
            self.beta = disease_vars['beta'] 
            self.k = disease_vars['recovery_rate']
        except Exception as e:
            raise Exception(f"Error parsing variables: {e.toString()}")

        self.connections = []

        self.S_travel = 0
        self.I_travel = 0
        self.R_travel = 0

    def increment(self):
        self.t += self.delta_t
        # Calculate the change for the two possible changes
        dN_SI = np.random.binomial(self.S, 1 - np.exp(-1 * self.beta * self.I / self.N * self.delta_t))
        dN_IR = np.random.binomial(self.I, 1 - np.exp(-1 * self.k * self.delta_t))

        for connection in self.connections:
            # TODO: make constant factor
            S_change = int(self.S * connection.S_amount)
            I_change = int(self.I * connection.I_amount)
            R_change = int(self.R * connection.R_amount)

            connection.node.add_incoming_travel(S_change, I_change, R_change)
            self.S -= S_change
            self.I -= I_change
            self.R -= R_change

        self.S -= dN_SI
        self.I += dN_SI - dN_IR 
        self.R += dN_IR

        if self.I > self.peak_I:
            self.peak_I = self.I

        self.total_I += dN_SI

        # Calculate 

        assert(self.S + self.I + self.R == self.N, "Error in updating, population does not equal SIR sum")
        return self.S, self.I, self.R

    def get_state(self):
        return self.S, self.I, self.R

    def get_header(self):
        return ["Susceptible", "Infected", "Recovered"]

    def add_travel_information(self, target_node, percent_travel):
        self.connections.append(Connection(destination_node=target_node, S_amount=percent_travel, I_amount=percent_travel, R_amount=percent_travel))

    def add_incoming_travel(self, S_amount, I_amount, R_amount):
        self.S_travel += S_amount
        self.I_travel += I_amount
        self.R_travel += R_amount

    def apply_travel(self):
        self.S += self.S_travel
        self.I += self.I_travel
        self.R += self.R_travel

        if self.I > self.peak_I:
            self.peak_I = self.I

        self.S_travel = 0
        self.I_travel = 0
        self.R_travel = 0

    def get_peak_I(self):
        return self.peak_I

    def get_cumulative(self):
        return self.total_I

class Connection:
    def __init__(self, destination_node, S_amount, I_amount, R_amount):
        self.node = destination_node 
        self.S_amount = S_amount
        self.I_amount = I_amount
        self.R_amount = R_amount
        # for name, value in zip(kwargs, kwargs.values()):
        #     self.name = value