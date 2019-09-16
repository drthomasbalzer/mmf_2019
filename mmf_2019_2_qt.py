################
## Author: Thomas Balzer
## (c) 2019
## Material for MMF Stochastic Analysis - Fall 2019
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist
import plot_utilities as pu


###########
##### Demo of Quantile Transformation
###########

## basic utility to create uniform sample

def getUniformSample(sz):

    lower_bound = 0.
    upper_bound = 1.

    sample = np.random.uniform(lower_bound, upper_bound, sz)

    return sample

## plot without transformation

def uniform_histogram(sz):

    uniform_sample = getUniformSample(sz)

    num_bins = 50
    hp = pu.PlotUtilities("Histogram of Uniform Sample of Size={0}".format(sz), 'Outcome', 'Rel. Occurrence')
    hp.plotHistogram(uniform_sample, num_bins)


#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def binomial_histogram(p, sz):

    sample = [dist.binomial_inverse_cdf(p, u) for u in getUniformSample(sz)]
    num_bins = 100

    hp = pu.PlotUtilities("Histogram of Binomial Sample with Success Probability={0}".format(p), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)


#####
## Create distribution via Quantile Transform -- $Exp(\lambda)$ distribution
#####

def exponential_histogram(_lambda, sz):

    sample = [dist.exponential_inverse_cdf(_lambda, u) for u in getUniformSample(sz)]
    num_bins = 50

    # the histogram of the data
    n, bins, _hist = plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Exponential Sample with Parameter={0}".format(_lambda))

    y = [dist.exponential_pdf(_lambda, b) for b in bins]

    plt.plot(bins, y, 'r--')
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def normal_histogram(mu, var, sz):

    nd = dist.NormalDistribution(mu, var)
    #######
    ### transform the uniform sample
    #######
    sample = [nd.inverse_cdf(u) for u in getUniformSample(sz)]
    num_bins = 60

    hp = pu.PlotUtilities("Histogram of Normal Sample with Mean={0}, Variance={1}".format(mean, var), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)


def lognormal_histogram(mu, var, sz):

    uniform_sample = getUniformSample(sz)

    nd = dist.NormalDistribution(0, variance)
    #######
    ### transform the uniform sample
    #######
    ###
    strike = 70.
    sample = [''] * 2
    sample[0] = [mean * np.exp(nd.inverse_cdf(u) - 0.5 * var) for u in uniform_sample]
    sample[1] = [max(s - strike, 0.) for s in sample[0]]
    num_bins = 75

    hp = pu.PlotUtilities("Histogram of Lognormal Sample with Mean={0}, Variance={1}".format(mu, var), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram([sample[0]], num_bins)


if __name__ == '__main__':

    calc_type = 3

    size = 50000

    if (calc_type == 0):  ### uniform sample
        uniform_histogram(size)
    elif (calc_type == 1):  ### generate a binomial distribution
        p = 0.40
        binomial_histogram(p, size)
    elif (calc_type == 2):  ### generate an exponential distribution
        _lambda = 1.
        exponential_histogram(_lambda, size)
    elif (calc_type == 3):  ### generate a normal distribution
        mean = 30.
        variance = 0.2
        normal_histogram(mean, variance, size)
    else:  ### generate a lognormal distribution
        mean = 100
        variance = 0.1
        lognormal_histogram(mean, variance, size)

