# data source: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data
#To get the current working directory use
#import os
#cwd = os.getcwd()
#os.chdir(path) ("change the current working directory to path")
#
import os
cwd = os.getcwd()
print(cwd)
os.chdir('D:\pycharm\data')
cwd1 = os.getcwd()
print(cwd1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #set plot style

#######################################################################################################
#Example 1: 02.04 president height
data = pd.read_csv('president_heights.csv')
print(data.head())

#extract heights as array
heights_df = data['height(cm)']
print(heights_df) # this is dataframe, not array!

heights = np.array(data['height(cm)'])
print(heights) #output as array

#summary statistics:
print('mean height is ', np.mean(heights)) #or: heights.mean()
print('std is', np.nanstd(heights))
print('min is', np.min(heights))
print('max is', np.min(heights))

print("25th percentile", np.percentile(heights, 25))
print("median", np.median(heights))
print("75th percentile", np.percentile(heights, 75))

#visualization
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number')
plt.show(); #notice the ; here!

##################################################################################################
#Example 2: 02.06 Counting Rainy days
rainfall_data = pd.read_csv('Seattle2014.csv')
print(rainfall_data.head())
rainfall = np.array(rainfall_data['PRCP'])
print (rainfall)
print(rainfall.shape) #so it's a yearly dataset

#the original code from the book:
rainfall_value = rainfall_data['PRCP'].values #another way to transform a dataframe into an array: .values
print (rainfall_value)
print(np.array_equal(rainfall,rainfall_value)) #test if two arrays are the same

#going on using rainfall
inches = rainfall/254.0 #transforming 1/10mm -> inches
print(inches.shape)

#visualization
plt.hist(inches,40, color= 'salmon')
plt.show();

#how many rainy days were there in the year?
print((inches>0).sum()) #count days with inches >0
print(np.count_nonzero(inches)) #same way
print(pd.value_counts(inches)) #not exactly what we what but gives our frequency of each value
print(pd.value_counts(inches>0)) #help us locate a range: TRUE 150, same answer as others

# What is the average precipitation on those rainy days?
print(np.mean(inches[inches>0]))

raindays = inches[inches>0] #create the new array that only contains rainday records # we can use boolean masks as well, as to be shown below
print(raindays.shape) #check if correct: 150 records
print(np.mean(raindays))

#cannot use np.mean(inches>0), which returns the mean of boolean value. See example as below:
a = np.array([1,1,0,0])
print(np.mean(a>0)) #it does not return (1+1)/2 = 1 instead returns 2/4 = 0.5

#How many days were there with more than half an inch of rain?
print(len(inches[inches>0.5]))

#to answer the above 3 question, refer to notes about how boolean values help

#show inches between 0.5 and 1
print(np.sum((inches > 0.5) & (inches < 1)))
print(np.sum(~( (inches <= 0.5) | (inches >= 1)))) #bitwise operator, ~ means !

#using boolean to create a mask (similar than what we did above
rainy = (inches > 0)

days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):",
      np.median(inches[rainy & ~summer]))
