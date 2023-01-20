from disease_node import DiseaseNode
import numpy as np
from numpy.random import binomial

class SEIRSNode(DiseaseNode):
    def __init__(self, total_population, disease_vars, delta_t = 1, name = '', start_with_disease=False):
        self.t = 0
        self.delta_t = delta_t
        self.S = total_population - 1000 if start_with_disease else total_population
        self.E = 0
        self.I = 1000 if start_with_disease else 0
        self.R = 0

        self.total_I = 0
        self.peak_I = 0

        self.name = name

        try:
            self.beta = disease_vars['beta']
            self.mu = disease_vars['birth_rate']
            self.mu_2 = disease_vars['natural_death_rate']
            self.v = disease_vars['disease_death_rate'] 
            self.sigma = disease_vars['incubation_rate']
            self.gamma = disease_vars['recovery_rate']
            self.omega = disease_vars['lost_immunity_rate']
            self.quarantine_buffer = [(0, 0)] * disease_vars.get("quarantine_days", 0)
            self.all_quarantine = disease_vars.get("all_quarantine", False)
            self.threshold_func = disease_vars['threshold_function']

        except Exception as e:
            raise Exception(f"Error parsing variables: {str(e)}")

        self.connections = []

        self.S_travel_in = 0
        self.E_travel_in = 0
        self.I_travel_in = 0
        self.R_travel_in = 0

        self.S_travel_out = 0
        self.E_travel_out = 0
        self.I_travel_out = 0
        self.R_travel_out = 0

    # Returns Suscetible, Exposed, Infected, Recovered at each timestamp
    def increment(self):
        self.t += self.delta_t
        dN_SE = binomial(self.S, 1 - np.exp(-1 * self.beta * self.I / (self.S + self.I + self.E + self.R) * self.delta_t))
        dN_EI = binomial(self.E, 1 - np.exp(-1 * self.sigma * self.delta_t))
        dN_IR = binomial(self.I, 1 - np.exp(-1 * self.gamma * self.delta_t))
        dN_RS = binomial(self.R, 1 - np.exp(-1 * self.omega * self.delta_t))
        dN_NS = binomial(self.S, 1 - np.exp(-1 * self.mu * self.delta_t))
        dN_SN = binomial(self.S, 1 - np.exp(-1 * self.mu_2 * self.delta_t))
        dN_EN = binomial(self.E, 1 - np.exp(-1 * self.mu_2 * self.delta_t))
        dN_IN = binomial(self.I, 1 - np.exp(-1 * (self.mu_2 + self.v) * self.delta_t))
        dN_RN = binomial(self.R, 1 - np.exp(-1 * self.mu_2 * self.delta_t))

        # Change in susceptible
        self.S += -dN_SE + dN_RS + dN_NS - dN_SN

        # Change in exposed
        self.E += dN_SE - dN_EI - dN_EN

        # Change in infected
        self.I += dN_EI - dN_IR - dN_IN
        self.total_I += dN_EI

        # Change in recovered
        self.R += dN_IR - dN_RS - dN_RN

        for connection in self.connections:
            travel_amount = connection.get_travel(self.threshold_func, self.I, self.get_population())
            S_change = int(self.S / self.get_population() * travel_amount)
            E_change = int(self.E / self.get_population() * travel_amount)
            I_change = int(self.I / self.get_population() * travel_amount)
            R_change = int(self.R / self.get_population() * travel_amount)

            connection.node.add_incoming_travel(S_change, E_change, I_change, R_change)
            self.S_travel_out -= S_change
            self.E_travel_out -= E_change
            self.I_travel_out -= I_change
            self.R_travel_out -= R_change

        return self.S, self.E, self.I, self.R

    def get_state(self):
        return self.S, self.E, self.I, self.R, self.get_population()

    def get_header(self):
        return ["Susceptible", "Exposed", "Infected", "Recovered"]

    def add_travel_information(self, nodes, target_node, travel_vars, dist):
        self.connections.append(travel_vars['connection'](name=self.name, nodes=nodes, destination_node=target_node, travel_vars=travel_vars, distance=dist))

    def add_incoming_travel(self, S_amount, E_amount, I_amount, R_amount):
        self.S_travel_in += S_amount
        self.E_travel_in += E_amount
        self.I_travel_in += I_amount
        self.R_travel_in += R_amount

    def apply_travel(self):
        if self.all_quarantine:
            # Currently assumes everyone will come out of quarantine either recovered or still suscetible
            self.quarantine_buffer.append((self.S_travel_in, self.E_travel_in + self.I_travel_in + self.R_travel_in))

        else:
            self.quarantine_buffer.append((0, self.I_travel_in))
            self.S += self.S_travel_in
            self.E += self.E_travel_in
            self.R += self.R_travel_in

        # Items in quarantine are saved as a tuple of susceptible and recovered patients
        new_S, new_R = self.quarantine_buffer.pop(0)
        self.S += new_S 
        self.I += new_R

        self.S += self.S_travel_out
        self.I += self.I_travel_out
        self.E += self.E_travel_out
        self.R += self.R_travel_out

        if self.I > self.peak_I:
            self.peak_I = self.I

        self.S_travel_in = 0
        self.E_travel_in = 0
        self.I_travel_in = 0
        self.R_travel_in = 0

        self.S_travel_out = 0
        self.E_travel_out = 0
        self.I_travel_out = 0
        self.R_travel_out = 0

    def get_peak_I(self):
        return self.peak_I

    def get_cumulative(self):
        return self.total_I

    def get_population(self):
        return self.S + self.I + self.E + self.R
