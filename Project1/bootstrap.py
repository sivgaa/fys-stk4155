# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
%matplotlib inline

import numpy as np
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt

# Returns mean of bootstrap samples 
# Bootstrap algorithm
def bootstrap(data, datapoints):
    t = np.zeros(datapoints)
    n = len(data)
    # non-parametric bootstrap         
    for i in range(datapoints):
        t[i] = np.mean(data[np.random.randint(0,n,n)])
    # analysis    
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(t),np.std(t)))
    return t

# We set the mean value to 100 and the standard deviation to 15
mu, sigma = 100, 15
datapoints = 5
# We generate random numbers according to the normal distribution
x = mu + sigma*np.random.randn(datapoints)
print(x)
# bootstrap returns the data sample                                    
t = bootstrap(x, datapoints)