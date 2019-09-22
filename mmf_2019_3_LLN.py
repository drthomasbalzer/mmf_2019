################
## Author: Thomas Balzer
## (c) 2019
## Material for MMF Stochastic Analysis - Fall 2019
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


###########
##
## Demo of the Law of Large Numbers
##
###########

def binomial_lln(sample_size, p):

    ######
    ## Step 1 - create sample of independent uniform random variables

    lower_bound = 0.
    upper_bound = 1.
    uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

    ######
    ## Step 2 - transform them to $B(1,p)$ distribution
    sample = [dist.binomial_inverse_cdf(p,u) for u in uni_sample]


    x_ax = [k for k in range(sample_size)] # values on the x axis
    n_plots = 2
    y_ax = [[0.] * sample_size for j in range(n_plots)]

    # y_values (1) - actual average
    y_ax[1] = [p for x in range(sample_size)]

    # y_values (0) - cumulative average of all the samples
    y_ax[0] = [sum([sample[j] for j in range(k+1)]) / (k+1) for k in range(sample_size)]

    mp = pu.PlotUtilities("Cumulative Average", 'x', 'Average')
    mp.multiPlot(x_ax, y_ax)


if __name__ == '__main__':

    sz = 1500
    p = .75
    binomial_lln(sz, p)



