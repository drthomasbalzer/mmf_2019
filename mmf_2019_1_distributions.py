################
## Author: Thomas Balzer
## (c) 2019
## Material for MMF Stochastic Analysis - Fall 2019
################

import plot_utilities as pu
import core_math_utilities as dist

###############
##
##  set up plots for pdf of multiple distributions
##
###############

def plotMultiDistributions(distrib, min_val, max_val, steps, chart_title):

       x_label = 'x'
       y_label = 'PDF Value'

       step = (max_val - min_val) / steps
       x_ax = [min_val + step * k for k in range(steps)]
       y_ax = [[dist.pdf(x_val) for x_val in x_ax] for dist in distrib]

       mp = pu.PlotUtilities(chart_title, x_label, y_label)
       mp.multiPlot(x_ax, y_ax)

if __name__ == '__main__':

    n_plots = 5

    versions = [k for k in range(1, 5)]

    for version in versions:
        title = ''
        if (version == 1):
            ### example of normal distributions with different $\sigma$ and $\mu = 0$.
            min_val = -5.
            max_val = 5.
            title = 'Normal Distribution with fixed $\mu$'
            d = [dist.NormalDistribution(0., 1. * (1. + float(k))) for k in range(n_plots)]
        elif (version == 2):
            ### example of normal distributions with different $\mu$ and $\sigma = 1$.
            min_val = -5.
            max_val = 5.
            title = 'Normal Distribution with fixed $\sigma$'
            d = [dist.NormalDistribution(-2. + float(k), 1.) for k in range(n_plots)]
        elif (version == 3):
            ## example of Exponential distributions
            min_val = 0.
            max_val = 4.
            title = 'Exponential Distribution'
            d = [dist.ExponentialDistribution(0.5 * (1. + float(k))) for k in range(n_plots)]
        elif (version == 4):
            min_val = 0.
            max_val = 5.
            title = 'Uniform Distribution'
            d = [dist.UniformDistribution(0., 1. + float(k)) for k in range(n_plots)]

        plotMultiDistributions(d, min_val, max_val, 1000, title)
