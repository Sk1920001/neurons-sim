import numpy as np

def normal_pdf(t, sigma):
    return (
        1 / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-(t ** 2) / (2 * sigma ** 2))
    )


def neuron_kernel(p, l, v):

    def k(t):

        sigma = l / 2

        return (
            p
            * normal_pdf(t, sigma)
            * np.sin(
                2 * np.pi * (t / l) ** v
            )
        )

    return k


def sigmoid(max_value, min_value, gain, offset):
    def s(t):
        return min_value + 2 * (max_value - min_value) / (1 + np.exp(-gain * (t - offset)))

    return s
