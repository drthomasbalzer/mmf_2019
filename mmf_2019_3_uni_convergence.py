################
## Author: Thomas Balzer
## (c) 2019
## Material for MMF Stochastic Analysis - Fall 2019
################


import numpy as np
import plot_utilities as pu

def uniform_histogram_powers(sz, powers):

    lower_bound = 0.
    upper_bound = 1.
    uniform_sample = np.random.uniform(lower_bound, upper_bound, sz)

    num_bins = 25
    samples = [[np.power(u, pow) for u in uniform_sample] for pow in powers]

    mp = pu.PlotUtilities("Histogram of Uniform Sample of Size={0}".format(sz), 'Outcome', 'Rel. Occurrence')
    mp.plotHistogram(samples, num_bins, powers)

if __name__ == '__main__':

    sz = 100000
    powers = [1.0, 1.25, 1.5, 2.]
    uniform_histogram_powers(sz, powers)


