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

#linear regression by least square 

slope , intercept = np.polyfit('x','y',1)# 3rd argument is the dimension of the polynomial

#drawing regression line on the plot 

_ = plt.plot('x', 'y' , marker = '.', linestype = 'none')
# draw scatter plot
_ = plt. margins(0.02)
_ = plt.xlabel('')
_ = plt.ylabel('')

slope, intercept = np.polyfit('x' , 'y', 1)

x = np.array([0,100])
y = slope * x + intercept 

_ = plt.plot(x,y)

plt.show()

#finding optimal slope for the data 

a_vals = np.linspace(0,0.1,200)
# return 200 numbers between 0 and 0.1

rss = np.empty_like(a_vals)

# return 200 array which have same datatype with a_vals

#compute sum of squares 

for i, a in enumerate(a_vals):
    rss[i] = np.sum(('y_data' - 'slope'*'x_data' - 'intercept')**2)
    
plt.plot(a_vals, rss, '-')
plt.xlabel('')
plt.ylabel('')
plt.show()

#iterate through and find slope and intercept for 4 different dataset 

anscombe_x = ['x1','x2','x3','x4']
anscombe_y = ['y1','y2','y3','y4']

for x, y in zip(anscombe_x, anscombe_y):
    a, b = np.polyfit(x,y,1)
    



