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
    
#Bootstraping 
    #The use of resample data to perform statistical inference
    
#Bootstrap Sample : A resampled array of data
#Bootstrap replicate : a statistic computed from the resampled data 
    
#To do the resample 
import numpy as np
np.random.choice([1,2,3,4,5], size = 5)


#visualizing bootstrap sample 

for i in range(50):
    bs_sample = np.random.choice('data',len('data'))
    
    x,y = ecdf(bs_sample)
    _ = plt.plot(x,y, marker = '.' , linestyle = 'none', alpha = 0.1, color = 'grey')
    #alpha can change the transparancy 

#draw ecdf for the real data 
    
x, y = ecdf('data')
_ = plt.plot(x,y, marker = '.')
plt.show()

#function to get boostrap replicates

def bootstrap_replicates_1D(data, func):
    bs_sample = np.random.choice(data,len(data))
    return func(bs_sample)

bs_replicates = np.empty(10000)

for i in range(10000):
    bs_replicates[i] = bootstrap_replicates_1D('data',np.mean)
    
_ = plt.plot(bs_replicates, bins = 30, nomred = True)

#normed makes total area of the bar equal to 1 so hist gets approximately pdf
plt.show()

#confidene interval of statistics : 
    #If we repeat the measurement again and agin, p% of observed values would lie within the p% of confidence interval

confidence_interval = np.percentile(bs_replicates,[2.5,97.5])

def draw_bootstrap_replicates(data, func, size =1):
    """Draw replicates from the bootrap samples"""
    bs_replicates = np.empty(size)
    
    for i in range(size):
        bs_replicates[i] = bootstrap_replicates_1D(data, func)
        
    return bs_replicates

#bootstrap estimates of the PDF of the mean 
    
draw_bootstrap_replicates('data', np.mean, size  = 10000)

#compute standard Error using std of real data 

sem = np.std('data') / np.sqrt(len('data'))

#compute standard deviation from bootstrap replicates 

bs_std = np.std(bs_replicates)

# and sem and bs_std has the same value 

#nonparatemetirc inference
    #Make no assumption about the model or probability distribution underlying the data


#Pair bootstrap 
    #bootstrap the indicies and get value using that indicies
#swing state data has two relevant variable one is total_votes and the other is dem_share
total_votes = np.empty(1000)
dem_share = np.empty(1000)
inds = np.arange(len(total_votes))
bs_inds = np.random.choice(inds, len(inds))
bs_total_votes = total_votes[bs_inds]
bs_dem_share = dem_share[bs_inds]

def draw_bs_pairs_linreg(x,y, size = 1 ):
    """perform pairs bootstrap for linear regression"""
    inds = np.arange(len(x))
    bs_slope_rep = np.empty(size = size)
    bs_intercept_rep = np.empty(size = size)
    
    for i in range(size):
        bs_inds = np.random.choice(inds,len(inds))
        bs_x = x[bs_inds]
        bs_y = y[bs_inds]
        bs_slope, bs_intercept = np.polyfit(bs_x, bs_y, 1)
        
    return bs_slope_rep, bs_intercept_rep

#plotting bootstrap regression


x = np.array([0,100])

#plot the regression lines

bs_slope_rep, bs_intercept_rep = draw_bs_pairs_linreg('x','y')
for i in range(100):
    plt.plot(x, bs_slope_rep[i]*x + bs_intercept_rep[i],linewidth = 0.5, alpha = 0.2, color = 'red')

#plot the data point
plt.plot('data', 'data', marker = '.' , linestyle = 'none')

#Formulating and Simulating hypothesis 
#Hypothesis test : Assessment of how reasonable the observed data are assuming a hypothesis is true. 
#Permutation : Random reordering or entries in an array 

#Try to figure out whether Pensilvania and Ohio has the same distribution. So mix combine two data and labeled again. 

import numpy as np 
dem_share_PA = np.empty(100)
dem_share_OH = np.emtpy(100)

dem_share_both = np.concatenate(dem_share_PA, dem_share_OH)
dem_share_perm = np.random.permutation(dem_share_both)
perm_sample_PA = dem_share_perm[:len(dem_share_PA)]
perm_sample_OH = dem_share_perm[len(dem_share_PA):]

#ou learned that permutation sampling is a great way to simulate 
#the hypothesis that two variables have identical probability distributions. 

def permutation_sample(data1, data2): 
    """Generate a permutation dataset from two seperate dataset"""
    """np.concatenate accept tuple as input so double braket"""
    
    data = np.concatenate((data1, data2))
    data_perm = np.random.permutation(data)
    
    data1_perm = data_perm[:len(data1)]
    data2_perm = data_perm[len(data1):]
    
    return data1_perm, data2_perm
    

#Visualizing permutation sample 
rain_june = np.empty(100)
rain_november = np.empty(100)
for i in range(50) : 
    sample_1 , sample_2 = permutation_sample((rain_june,rain_november))
    x_1 , y_1 = ecdf(sample_1)
    x_2 , y_2 = ecdf(sample_2)
    
    _ = plt.plot(x_1, y_1, marker = '.', linestyle = 'none' , color = 'red')
    _ = plt.plot(x_2, y_2, marker = '.', linestyle = 'none' , color = 'blue')
    
''' Visualizing permutation samples'''

x_1 , y_1 = ecdf(rain_june)
x_2 , y_2 = ecdf(rain_november)

_ = plt.plot(x_1, y_1, marker = '.', linestyle = 'none' , color = 'red')
_ = plt.plot(x_2, y_2, marker = '.' , linestyle = 'none', color = 'blue')
'''Visualizing original samples'''

plt.show()

#Test Statistic : A single statistic that can be computed from observed data and from data that you simulate under 
# the null hypothesis

# In this case, test statistic would be the difference between the PA and OH. 

#Permutation Replicate

np.mean(perm_sample_PA) - np.mean(perm_sample_OH)

#original data 

np.mean(dem_share_PA) - np.mean(dem_share_OH)

#permutation으로 눌 하이퍼시시가 참이라고 가정하고 만들어 낸 test statistic이 np.mean(perm_sample_PA) - np.mean(perm_sample_OH)
#인것이고, 이 permutation을 여러번해서 얻어낸 test statistic으로 만들어진 distribution 에서 original data에서 만들어낸 
#np.mean(dem_share_PA) - np.mean(dem_share_OH) value 보다 큰 부분의 확률은 p-value 

""" P-value is the probability of obtatining a value of your test statistic 
that is at least as extreme as what was observed, under the assumption the null 
hypothesis is true"""

"Null hypothesis significance Testing (NHST)"

