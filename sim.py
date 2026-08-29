from classes import Function 
from classes import Convolution 
from functions import neuron_kernel
import numpy as np

DELTA_T = 0.01


def main():
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

    stimulus = Function("Example Stimulus", functions=functions, delta_t=DELTA_T)
    stimulus.graph()

    # ON-fast-sustained
    p = 1
    l = 0.4
    v = 1.2

    neuron = Function(
        name="ON-fast-sustained",
        functions=[
            (2, neuron_kernel(p, l, v)) #2 seconds so the array dosen't get crazy long 
        ],
        delta_t= DELTA_T
    )
    neuron.graph()

    convolution = Convolution(stimulus, neuron)
    convolution.graph(DELTA_T)


if __name__ == "__main__":
    main()

        


