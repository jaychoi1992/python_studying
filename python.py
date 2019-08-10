# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

x = np.soort('variable')
y = np.arrange(1, len(x)+1) / len(x)
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

