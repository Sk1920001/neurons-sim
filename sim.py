from classes import Function, Convolution, PoissonProcess
from functions import neuron_kernel,sigmoid
import numpy as np

DELTA_T = 0.01



def main():
    x_values = np.arange(0, 21.5, DELTA_T)
    functions = [
                    (1.5, lambda x: -1),
                    (2, lambda x: 1),
                    (2, lambda x: -1),
                    (2, lambda x: 0),
                    (5, lambda x: np.sin(np.pi * np.power(x,2))),
                    (2, lambda x: 0),
                    (5, lambda x: 0.2 * x * np.sin(3 * np.pi * x)),
                    (2, lambda x: 0)
                ]

    stimulus = Function(functions=functions, x_values=x_values)
    # ON-fast-sustained
    p = 1
    l = 0.4
    v = 1.2

    neuron = Function(
        functions=[
            (2, neuron_kernel(p, l, v)) #2 seconds so the array dosen't get crazy long 
        ],
        x_values=np.arange(0, 2, DELTA_T)
    )
    #neuron.graph()

    convolution = Convolution(stimulus, neuron, min_max=True, delta_t=DELTA_T)
    convolution.graph()

    rate = Function(
        functions=[
            (round(np.max(convolution.y_values) - np.min(convolution.y_values),1), sigmoid(max_value=100, min_value=0.5, gain=4, offset=1))
        ],
        x_values=convolution.y_values
    )

    rate.graph(name="Firing Rate", x_label="Time(s)", y_label="Firing Rate (Hz)", alt_graph_x_values=x_values)

    points = PoissonProcess(rate_values=rate.y_values, x_values=x_values)
    points.graph(x_label="Time(s)", y_label="Count")



if __name__ == "__main__":
    main()

        


