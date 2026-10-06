import numpy as np
import matplotlib.pyplot as plt
from functions import generate_arrivals_poisson,generarte_poisson_process
import random 

class Function:
    def __init__(self, x_values=np.arange(0, 10, 0.01), functions = [(1, lambda x: 1)]) :   #The first value of the tuple is the length of the interval, the second value is the function to be applied in that interval

        self.functions= functions 
        self.x_values = x_values 
        self.y_values = np.zeros_like(self.x_values)
        self._evaluate()

    def _evaluate(self):
        if self.x_values[-1] > sum([value[0] for value in self.functions ]):
            raise ValueError(f"x_values exceed the total interval length. Please adjust x_values or the function intervals.")

        start = self.x_values[0]
        for interval, func in self.functions:
            mask = (self.x_values >= start) & (self.x_values < start + interval)
            self.y_values[mask] = func(self.x_values[mask] - start)
            start += interval


    def graph(self, name="Function", x_label="Time", y_label="Amplitude", alt_graph_x_values=None, color='blue'):
        x_values = self.x_values if alt_graph_x_values is None else alt_graph_x_values
        plt.plot(x_values, self.y_values[: x_values.shape[0]], c=color)
        plt.title(name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


        plt.show()
        plt.close()

class Convolution:
    def __init__(self, function1, function2, delta_t=0.01, min_max=False):
        self.function1 = function1
        self.function2 = function2
        self.delta_t = delta_t
        self.x_values = np.arange(0, 1, 0.1)
        self.y_values = np.arange(0, 1, 0.1)
        self.min_max = min_max
        self.evaluate()


    def evaluate(self):
        y_values = np.convolve(self.function1.y_values, self.function2.y_values) * self.delta_t 
        self.x_values = np.arange(y_values.shape[0]) * self.delta_t 

        if self.min_max:
            self.y_values = ((y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))) * 2 - 1
            return

        self.y_values = y_values[: self.x_values.shape[0]]


    def graph(self, title="Convolution", x_label="Time", y_label="Amplitude", color='blue'):
        plt.plot(self.x_values, self.y_values, c=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
        plt.close()

class PoissonProcess:
    def __init__(self, rate_values , max_time, time_delta):
        self.time_delta = time_delta
        self.rate_values = rate_values 
        self.rate_sup = np.max(rate_values)
        self.initial_events = generarte_poisson_process(rate=self.rate_sup * max_time)
        self.initial_arrivals = generate_arrivals_poisson(num=self.initial_events, max_time=max_time)
        self.accepted_arrivals = [] 
        self.evaluate()

    def evaluate(self):
        # Generate Poisson process based on the rate function
        for i in range(self.initial_arrivals.shape[0]):
            U = random.random()
            time_index = round(self.initial_arrivals[i]/self.time_delta)
            if U <= self.rate_values[time_index]/self.rate_sup :  #Uniform(0,1) CDF
                self.accepted_arrivals.append(self.initial_arrivals[i])
        self.accepted_arrivals = np.array(self.accepted_arrivals)




    def graph(self, title="Poisson Process", x_label="Time", y_label="Count", color='blue'):
        plt.scatter(self.accepted_arrivals , np.zeros_like(self.accepted_arrivals), c=color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.yticks([])

        plt.show()
        plt.close()
