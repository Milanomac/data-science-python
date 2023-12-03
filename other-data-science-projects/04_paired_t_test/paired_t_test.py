# In some cases, you might be interested in testing differences between samples of the same group 
# at different points in time. For instance, a hospital might want to test whether a weight-loss drug 
# works by checking the weights of the same group patients before and after treatment. 
# A paired t-test lets you check whether the means of samples from the same group differ.

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(11)

before = stats.norm.rvs(scale=30, loc=250, size=100)

after = before + stats.norm.rvs(scale=5, loc=-1.25, size=100)

weight_df = pd.DataFrame({"weight_before":before,
                          "weight_after":after,
                          "weifht_change":after-before})

print(weight_df.describe())            #check a summary of the data

# print(weight_df)

# Conducting a paired t-test to see whether this difference is significant at a 95% confidence level
t_test = stats.ttest_rel(a = before,
                         b = after)

print(t_test)
