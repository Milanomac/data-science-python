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

t_test = stats.ttest_rel(a = before,
                         b = after)

print(t_test)