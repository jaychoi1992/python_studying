


# Exploratory Data Analysis 

#1 histogram 
import matplotlib.pyplot as plt 
import seaborn as sns 
# the package which enhance the visual design

sns.set()
bin_edges = [0,10,100]
_ = plt.hist('variable' , bins = bin_edges)
#_  is for prevent unwanted graphs
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()

#2 sworm plt

_ = sns.swarmplot(x= '', y = '' , data = '')
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()

#3 Empirical CDF (ECDF)

import numpy as np 


x = np.sort('variable')
y = np.arange(1, len(x)+1) / len(x)


_ = plt.plot(x, y, marker = '.', linestyle = 'none')
_ = plt.xlabel('')
_ = plt.ylabel('')
_ = plt.margins(0.02)

plt.legend(('variable','variable','variable'),loc = 'lower right')
plt.show()

def ecdf (data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y


#Summary Statistics 
    
#1 calculating mean 
    
import numpy as np 

np.mean('variable')

#2 calculating percentile 
percentiles = np.array(25,50,75)

ptiles_vers = np.percentile('variable', [25,50,75])
#get output as arrary(['','',''])

#3 making boxplot 

import matplotlib.pyplot as plt 
import seaborn as sns 

_ = sns.boxplot(x = '', y = '', data = '')
_ = sns.xlabel('')
_ = sns.ylabel('')
plt.show()

#4 marking percentiles on the ecdf graph 

import numpy as np 

x = np.sort('variable')
y = np.arange(1,len(x)+1) / len(x)
_ = plt.plot(x,y, marker = '.',linestyle = 'none')
_ = plt.xlabel('')
_ = plt.ylabel('')
_ = plt.margins(0.02)

_ = plt.plot(ptiles_vers , percentiles , marker = 'D', color = 'red', linestyle = 'none')

plt.show()

# variance and Standard Deviantion 

#variance 

import numpy as np 

np.var('variable')

# Standard deviationo 
import numpy as np 
 
np.std('variable')
 
 #generate scatter plot 
 
import matplotlib.pyplot as plt 
_ = plt.plot( '', '', marker = '.', linestyle = 'none')
_ = plt.xlabel('')
_ = plt.ylabel('')

# covariance 

import numpy as np 

covariance_matrix = np.cov('variable','variable')

#[0,0] stands for variance for x, [1,1] for variance of y and [0,1],[1,0] is covariance. 

#Pearson correlation coefficient 

def Pearson_r (x, y):
   corr_mat =  np.corrcoef(x,y)
    
   return corr_mat[0,1]

#[0,0] and [1,1] should be 1 and [0,1],[1,0] is correlation coefficient
    
#Probabilities logic and Statistical Inference 
   
# simulating with 4 coins with head up as true 
   
n_all_heads = 0

for _ in range(1000):
   heads =  np.random.random(size = 4) < 0.5
   n_heads = np.sum(heads)
   if n_heads == 4 :
        n_all_heads += 1
        
n_all_heads / 1000

#Try to understand the equality of random function 
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)
random = np.empty(100000)

for i in range(100000):
    random[i] = np.random.random()
    
plt.hist(random)

plt.show()

#Bernoulli Trial

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    
    n_success = 0
    
    for i in range(n):
        random_number = np.random.random()
        if random_number < p:
            n_success += 1
            
#example : make loan with 0.05% default and do 1000 times simulation 
            
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)

result = np.empty(1000)

for i in range(1000):
    result[i] = perform_bernoulli_trials(100,0.05)
    
_ = plt.hist(result, normed = True)
# normed = True,so that the height of the bars of the histogram indicate the probability.

_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

#Probability Mass Functions(PMF)

# sampling from the binomial distribution 

import numpy as np 

np.random.binomial(n = 4, p = 0.4, size = 10)

#to plot the PMF we change bins edge and will use histogram

bins = np.arange(min('variable'), max('variable') + 1.5) - 0.5

_ = plt.plot('variable' , normed = True, bins = bins)
_ = plt.xlabel('')
_ = plt.ylabel('')
plt.show()
#Poisson Process : 
    #The timing of the next event is completely independent of wen the previous event happened.

#Poisson distribution :
    #The number r of arrivals oof a Poisson process in a given time interval with average rate of ramda arrivals
    #per interval is Pissoon distributed

# Limit of the Binomial distribution for low probability of success and large number of trials : rare event
    
#compare the mean and probability between Poisson distribution and Binomial Distribution 
import numpy as np
sample_poisson = np.random.poisson(10, size = 10000)
print('Poisson: ', np.mean(sample_poisson), np.std(sample_poisson))

# make 3 pairs of n and p which makes 10

n = [20,100,1000]
p = [0.5,0.1,0.01]

for i in range(3):
    sample_binomial = np.random.binomial(n[i],p[i],size = 10000)
    
    print('binomial distribution with n= ',n[i], ' ',p[i],' ', np.mean(sample_binomial), np.std(sample_binomial))
    

# the mean and std getting closser to poisson dist as p getting smaller and n getting bigger


#Probability Density Function
    
    

