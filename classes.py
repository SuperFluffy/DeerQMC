class SimulationParameters:
    def __init__(self,dictionary):
        d = dictionary

        self.beta  = d['beta']
        self.idtau = d['idtau']
        self.tn = d['tn']
        self.tnn = d['tnn']
        self.u = d['U']
        self.mu = d['mu']
        self.b = d['B']

        self.thermalization_steps = d['thermalizationSteps']
        self.measurements_steps = d['measurementSteps']

        self.x = d['x']
        self.y = d['y']
        self.n = d['N']

        self.lambda2 = d['lambda2 dictionary']
        self.nodes = d['nodes']

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self,other):
        return self.__dict__ == other.__dict__
