from abc import ABC, abstractmethod 

class DiseaseNode(ABC):
    
    @abstractmethod
    def increment(self):
        pass

    def get_time(self):
        return self.t

    @abstractmethod 
    def get_state(self):
        pass

    @abstractmethod 
    def get_header(self):
        pass

    @abstractmethod
    def apply_travel(self):
        pass 

    @abstractmethod 
    def add_incoming_travel(self, **kwargs):
        pass

    def get_peak_I(self):
        return self.peak_I

    def get_cumulative(self):
        return self.total_I
