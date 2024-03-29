################
## Author: Thomas Balzer
## (c) 2019
## Material for MMF Stochastic Analysis - Fall 2019
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def random_sample_sym_binomial(_p, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)
    sample = [dist.symmetric_binomial_inverse_cdf(_p, u) for u in uni_sample]
    return sample

def random_sample_normal(_vol, sz):

    uni_sample = np.random.uniform(0., 1., sz)
    sample = [_vol * dist.standard_normal_inverse_cdf(u) for u in uni_sample]
    return sample

def random_walk(_p, _steps, _paths, scaling):

    scaled_steps = int(_steps * scaling)
    samples = [random_sample_sym_binomial(_p, scaled_steps) for k in range(_paths)]

    x = [float(k / float(scaling)) for k in range(scaled_steps + 1)]
    paths = [ [sum(sample[0:k]) / np.sqrt(scaling) for k in range(len(sample) + 1)] for sample in samples ]
    mp = pu.PlotUtilities("Paths of Random Walk with Probability={0}".format(p), 'Time', 'Random Walk Value')
    mp.multiPlot(x, paths)

def standard_brownian_motion(_steps, _paths, scaling):

    scaled_steps = int(_steps * scaling)
    samples = [random_sample_normal(np.sqrt(1. / scaling), scaled_steps) for k in range(_paths)]

    x = [float(k / float(scaling)) for k in range(scaled_steps + 1)]
    paths = [ [sum(sample[0:k]) for k in range(len(sample) + 1)] for sample in samples ]
    mp = pu.PlotUtilities('Paths of Standard Brownian Motion', 'Time', 'Random Walk Value')
    mp.multiPlot(x, paths)

def random_walk_terminal_histogram(_p, _steps, _paths, scaling):

    scaled_steps = _steps * scaling
    samples = [random_sample_sym_binomial(_p, scaled_steps) for k in range(_paths)]

    terminal_value = [ sum([s for s in sample]) / np.sqrt(scaling) for sample in samples]

    mp = pu.PlotUtilities("Distribution of Terminal Value of Random Walk with Probability={0}".format(p), 'Value', 'Rel. Occurrence')
    mp.plotHistogram(terminal_value, 21)



if __name__ == '__main__':

    p = 0.5
    _paths = 10
    _steps = 10
    scaling = 100
    #random_walk(p, _steps, _paths, scaling)
    # random_walk_terminal_histogram(p, _steps, _paths, scaling)
    standard_brownian_motion(_steps, _paths, scaling)

