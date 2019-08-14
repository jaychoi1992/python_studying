#Optimal Parameter

#Parameter values that bring model in closest agreement with the data 


#exponential distribution 

import numpy as np 
import pyplotlib.pyplot as plt

tau = np.mean('data')
inter_nohitter_time = np.random.exponential(tau, 100000)

_ = plt.hist(inter_nohitter_time, bins = 50, normed = True, histtype = 'step')
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()

#overlay the theoratical CDF with the ECDF from the data will help you whether exponential distribution helps 
# you to understand the data. 

def ecdf (data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y

x, y = ecdf('real data')
x_theor, y_theor = ecdf(inter_nohitter_time)
# theoratically maden sample

_ = plt.plot(x,y)
_ = plt.plot(x_theor,y_theor)
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()

#compare with the other tau model, 

samples_half = np.random.exponential(tau/2, 10000)
samples_double = np.random.exponential(2*tau, 10000)

x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)
plt.show()

#based on the graph, the tau is the optimal parameter which can best match to the data. 




