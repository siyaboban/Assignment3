# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:54:23 2023

@author: siya
"""

import pandas as pd 
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

#tracing the indicator id and the country names with respect to the country codes
Countries = ['Australia','China','Canada','France','Germany','India','Japan','Spain','Mexico','United Kingdom']
             

Country_code = ['AUS','CHN','CAN','FRA','DEU','IND','JPN','ESP','MEX','GBR']
              

Country_names = {'AUS':'Australia','CHN':'China','CAN':'Canada','FRA':'France','DEU':'Germany','IND':'India','JPN':'Japan','ESP':'Spain','MEX':'Mexico','GBR':'United Kingdom'}

#taking the appropriate datasets
wb.series.info('EN.ATM.CO2E.KT')
#dataset with data regarding the total CO2 emissions in kiloton from a period of year 2010-2019
df_CO2 = pd.DataFrame(wb.data.DataFrame('EN.ATM.CO2E.KT',Country_code, time=range(2010, 2019)))
print(df_CO2)

#dataset with data regarding the total population from a period of year 2010-2019
wb.series.info('SP.POP.TOTL')

df_Pop = pd.DataFrame(wb.data.DataFrame('SP.POP.TOTL',Country_code, time=range(2010, 2019)))
print(df_Pop)

#transposed data of the total population dataset
df_trans = df_Pop.T
print(df_trans)

#pair plot comparing the two datasets
df_1 = pd.DataFrame()
CO2 = []
Population = []
for i in df_Pop:
    Population.extend(df_Pop[i])

for i in df_CO2:
    CO2.extend(df_CO2[i])

df_1['CO2'] = CO2
df_1['Population'] = Population

type(df_1)
sns.pairplot(df_1[['Population','CO2']])

'''function for finding K means clusttering'''
kmeans1 = KMeans(n_clusters=3, random_state=0).fit(df_1[['Population','CO2']])
kmeans1.inertia_
kmeans1.cluster_centers_
df_1['cluster'] = kmeans1.labels_
'''cluster found out'''
print(df_1)

'''plot for K means clusttering before normalisation'''
sns.scatterplot(x = 'CO2', y = 'Population' , hue='cluster', data = df_1)
plt.title("K-Means before normalisation")
plt.show()


from sklearn import preprocessing

data_1 = df_1.drop(['cluster'], axis = 1)

'''function called for clusttering'''
names = ['CO2','Population']
a = preprocessing.normalize(data_1, axis=0)
data_aft = pd.DataFrame(a,columns=names)
data_aft

kmeans2 = KMeans(n_clusters=3, random_state=0).fit(data_aft[['Population','CO2']])

kmeans2.inertia_

kmeans2.cluster_centers_

data_aft['cluster'] = kmeans2.labels_

'''cluster shown along the data'''
data_aft

'''plot for K means clusttering after normalisation'''
sns.scatterplot(x = 'CO2', y = 'Population' , hue='cluster', data = data_aft)
plt.title("K-Means after normalisation")
plt.show()

'''function to calculate the error limits'''
def func(x,a,b,c):
    return a * np.exp(-(x-b)**2 / c)


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

#dataset containing data regarding total electricity access
data_acs = wb.data.DataFrame('EG.ELC.ACCS.ZS',Country_code, time=range(2005, 2014))
data_acs

#dataset containing data regarding total energy consumption
data_con = wb.data.DataFrame('EG.USE.ELEC.KH.PC',Country_code, time=range(2005, 2014))
data_con

data_ele = pd.DataFrame()
access = []
consumption = []
for i in data_con:
    consumption.extend(data_acs[i])

for i in data_acs:
    access.extend(data_con[i])

data_ele['consumption'] = consumption
data_ele['access'] = access

'''plot for scattering'''
plt.scatter(data_ele['consumption'],data_ele['access'])
plt.title('Scatter plot before curve fitting')
plt.ylabel('Electric Power Consumption')
plt.xlabel('Access to electricity')
plt.show()

'''adding an exponential function'''
def expoFunc(x,a,b):
    return a**(x+b)

x_data = data_ele['consumption']
y_data = data_ele['access']
popt, pcov = curve_fit(expoFunc,x_data,y_data)

popt
pcov

a_opt, b_opt = popt
x_mod = np.linspace(min(x_data),max(x_data),100)
y_mod = expoFunc(x_mod,a_opt,b_opt)

'''plot for scattering after fitting the curve'''
plt.scatter(x_data,y_data)
plt.plot(x_mod,y_mod,color = 'red')
plt.title('Scatter plot after curve fitting')
plt.ylabel('Electric Power Consumption')
plt.xlabel('Access to electricity')
plt.savefig("curvefit_after.png")
plt.show()





