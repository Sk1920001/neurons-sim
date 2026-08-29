import numpy as np
import matplotlib.pyplot as plt

class Function:
    def __init__(self, name, delta_t=0.01, x_label="Time", y_label="Amplitude", functions = [(1, lambda x: 1)]) :   #The first value of the tuple is the length of the interval, the second value is the function to be applied in that interval

        self.name = name
        self.functions= functions 
        self.x_interval_lenght = sum([value[0] for value in functions ])
        self.x_label = x_label
        self.y_label = y_label
        self.x_values = np.arange(0, self.x_interval_lenght, delta_t)
        self.y_values = np.zeros_like(self.x_values)
        self._evaluate()

    def _evaluate(self):
        start = 0
        for interval, func in self.functions:
            mask = (self.x_values >= start) & (self.x_values < start + interval)
            self.y_values[mask] = func(self.x_values[mask] - start)
            start += interval


    def graph(self):
        plt.plot(self.x_values, self.y_values)
        plt.title(self.name)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')


        plt.show()
        plt.close()

class Convolution:
    def __init__(self, function1, function2):
        self.function1 = function1
        self.function2 = function2
    def graph(self, delta_t):

        y_values = np.convolve(self.function1.y_values, self.function2.y_values) * delta_t 
        x_values = np.arange(len(y_values)) * delta_t 


        plt.plot(x_values, y_values)
        plt.title(f'Convolution of {self.function1.name} and {self.function2.name}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')

        plt.show()
        plt.close()




