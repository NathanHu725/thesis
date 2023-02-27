from disease_node import DiseaseNode
import numpy as np
from numpy.random import binomial

class SEIRSDENode(DiseaseNode):
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
            self.beta_fun = disease_vars['beta_fun']
            self.mu = disease_vars['birth_rate']
            self.mu_2 = disease_vars['natural_death_rate']
            self.v = disease_vars['disease_death_rate'] 
            self.sigma = disease_vars['incubation_rate']
            self.gamma = disease_vars['recovery_rate']
            self.omega = disease_vars['lost_immunity_rate']
            self.quarantine_days = disease_vars.get("quarantine_days", 0)
            self.quarantine_buffer = [(0, 0, 0, 0)] * self.quarantine_days
            self.all_quarantine = disease_vars.get("all_quarantine", False)
            self.threshold_func = disease_vars['threshold_function']
            self.testing_func = disease_vars['testing_function']

        except Exception as e:
            raise Exception(f"Error parsing variables: {str(e)}")

    # Returns Suscetible, Exposed, Infected, Recovered at each timestamp
    def increment(self):
        self.t += self.delta_t
        dN_SE = self.beta_fun(self.t, self.beta) * max(self.S, 0) * max(self.I, 0) * self.delta_t / (max((self.S + self.E + self.I + self.R), 0))
        dN_EI = max(self.E, 0) * self.sigma * self.delta_t
        dN_IR = max(self.I, 0) * self.gamma * self.delta_t
        dN_RS = max(self.R, 0) * self.omega * self.delta_t
        dN_NS = max((self.S + self.E + self.I + self.R), 0) * self.mu * self.delta_t
        dN_SN = max(self.S, 0) * self.mu_2 * self.delta_t
        dN_EN = max(self.E, 0) * self.mu_2 * self.delta_t
        dN_IN = max(self.I, 0) * (self.mu_2 + self.v) * self.delta_t
        dN_RN = max(self.R, 0) * self.mu_2 * self.delta_t

        # Change in susceptible
        self.S += -dN_SE + dN_RS + dN_NS - dN_SN

        # Change in exposed
        self.E += dN_SE - dN_EI - dN_EN

        # Change in infected
        self.I += dN_EI - dN_IR - dN_IN
        self.total_I += dN_EI

        # Change in recovered
        self.R += dN_IR - dN_RS - dN_RN

        return self.S, self.E, self.I, self.R
    
    def add_travel_information(self, nodes, target_node, travel_vars, dist):
        return 0

    def add_incoming_travel(self, S_amount, E_amount, I_amount, R_amount):
        return 0

    def apply_travel(self):
        return 0

    def get_state(self):
        return self.S, self.E, self.I, self.R, self.get_population()

    def get_header(self):
        return ["Susceptible", "Exposed", "Infected", "Recovered"]

    def get_peak_I(self):
        return self.peak_I

    def get_cumulative(self):
        return self.total_I

    def get_population(self):
        return self.S + self.I + self.E + self.R
