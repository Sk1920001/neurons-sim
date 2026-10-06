import numpy as np
import random 
import math
from scipy.stats import poisson

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


def generarte_poisson_process(rate):
    prob = random.random() #U(0,1)
    cont = 0
    sum = 0
    while True:
        sum +=  poisson.pmf(cont, rate)
        if sum >= prob:
            return cont
        cont+=1

def generate_arrivals_poisson(num,max_time):
    times= np.zeros(num)
    for i in range(num):
        times[i] = np.random.uniform(0,max_time)
    times.sort()
    return times


