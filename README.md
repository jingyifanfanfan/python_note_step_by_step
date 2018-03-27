# python_note_step_by_step
Learning notes of python

# 1. Python Data science handbook.py
The codes are essentially from https://jakevdp.github.io/PythonDataScienceHandbook/, which I myself learn/refresh my memory of using Python by practicing all the example codes. The codes should later be able to serve as a notebook for reference when using _Numpy_ (array manipulation), _Pandas_, _matplotlib_ and _Scipy_.

Based on the original codes, I sometimes twist them a bit to help better understanding (for myself), so for any reader (if there will be any), there will be some discrepencies from the handbook. Usually I just improvise some mini-example for myself to help me really understand a function. For example:

 ``python
# a small example here: binning data
np.random.seed(42)
x = np.random.randn(100) # an array of 100 elements of standard distribution (0,1); 100 here can also be multidimentional, e.g.[2,4]

bins = np.linspace(-5,5, 20) # create 20 numbers between -5 and 5
counts = np.zeros_like(bins) # Return an array of zeros with the same shape and type as a given array.

print(bins)
print(counts)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)   #see below
# Explanation: -> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
# -> array([0, 5, 1, 2]), for [-10, 10, 2, 3], -10 is even before 1; 10 is more than 5, 2 is between 1-2 (the first bin)... default  'side = 'left''

print(type(i))
np.add.at(counts, i, 1)   # add 1 to each of the bins when they appear
#  very important note here: if assign counts1 = np.add.at(counts,i,1) will return nothing since it does not store results;
#  similar to when writing a loop you will use return but not print, otherwise it gives you nothing
#  simply print(counts)

print(counts)

# plot the results
import matplotlib.pyplot as plt
plt.plot(bins,counts,linestyle = 'steps')
plt.show();

plt.hist(counts,bins)
plt.show();

plt.hist(x, bins, histtype='step')
plt.show();  #see the difference; we will dive deeper into visualization later``

In the above chunk of code, comments such like _Explanation_, _very important notes_ or the other 2 plots following the first one are just notes/trial for me.

Everything in the file is by typing, which I found being vastly different from reading and copying. Although these are really simple and basic codes, I believe that it is a great start as well a solid foundation to digest more sophisticated knowledge. Hopefully anyone reading it would found it helpful as well.

# 2. Python Data science handbook example.py

In the handbook there are some examples with data. I simply put them in a separate file from the notes for more structured documentation. 

Data available here: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data




<currently updated before: https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html>
