


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

_ = plt.plot(ptiles_vers , percntiles , marker = 'D', color = 'red', linestyle = 'none')

plt.show()






