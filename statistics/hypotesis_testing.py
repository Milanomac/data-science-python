# DESCRIPTION:      This script demonstrates hypothesis testing principle in Python.
#                   One sample and two sample t-test is performed for populations following poisson distribution.

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Generating two random pupulations with poisson.rvs
np.random.seed(6)

population_ages1 = stats.poisson.rvs(loc=18, mu=35, size = 150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size = 100000)
population_ages = np.concatenate((population_ages1, population_ages2))

minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

print(f'The population mean is: {population_ages.mean()}')
print(f'The mean of ages in Minnesota is: {minnesota_ages.mean()}')

# T-test
t_test = stats.ttest_1samp(a = minnesota_ages,              # Sample data
                          popmean = population_ages.mean()) # Population mean

print(f'The test statistic and p-value are:\n{t_test}')

# Check quantiles for 95%
lower_quantile = stats.t.ppf(q=0.025, df=49)
upper_quantile = stats.t.ppf(q=0.975, df=49)

print(f'The lower quantile is {lower_quantile}. Since our t-statistic = -2.573 is below the lower quantile we reject the null hypothesis.')

# Check only the p-value
p_val = stats.t.cdf(x = -2.5742, df = 49)* 2 #two tailed test

# Creating confidence interval
sigma = minnesota_ages.std()/math.sqrt(50) # Sample stdev/sample size

conf_interv = stats.t.interval(0.95,                          # Confidence level
                               df = 49,                       # Degrees of freedom
                               loc = minnesota_ages.mean(),   # Sample mean
                               scale = sigma)                 # Standard dev estimate
print(conf_interv)

# Two sample T-test
np.random.seed(12)
wisconsin_ages1 = stats.poisson.rvs(loc=18, mu=33, size=30)
wisconsin_ages2 = stats.poisson.rvs(loc=18, mu=13, size=20)
wisconsin_ages = np.concatenate((wisconsin_ages1, wisconsin_ages2))

print(wisconsin_ages.mean())
print(minnesota_ages.mean())

two_sample_ttest = stats.ttest_ind(a = minnesota_ages,
                                   b = wisconsin_ages,
                                   equal_var = False) #samples do not have equal variance
print(two_sample_ttest)
# pvalue=0.09073104343957748
