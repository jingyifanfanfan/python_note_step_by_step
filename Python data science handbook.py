1. Data type in Python

L = list(range(10))
print(L)
print(type(L))
print(type(L[1]))

L2 = [str(c) for c in L]
print(L2)
print(type(L2))
print(type(L2[0]))

L3 = [True, 0.3, "2", 3.33]
# print all type:
for item in L3:
    print(type(item))

#print all types in one command:
print([type(item) for item in L3])


import array
A = array.array("i",L)
# i here indicates that contents in L are integers; wont be printed out
print(A)
print(A[0])
print(type(A[0]))

#for comparison:
A1 = array.array(L)
#returns: TypeError: array() argument 1 must be a unicode character, not list
# check for more info in Powershell: import array, array.array?

######################################################################################################################
#2. create an array
import numpy as np

#create an integer array:
array1 = np.array([1,3,4,2])
array2 = np.array([2,3.3,4,1,3])
#check difference between array 1& array 2; array 2 are uniformed as float
print(array1, array2)
print(type(array2[0]))

#then to explicitly set the data type of the array, use dtype = :
array3 = np.array([1,2,3,4], dtype = 'float32')
print(array3)

#creating MULTI-DIMENTIONAL ARRAY
array_multi = np.array([range(i, i+3)for i in [2,4,6]])
print(array_multi)

#creating different arrays:
# all-0 array
array_0 = np.zeros(10,dtype =int)
print(array_0)

#all-1 array/matrix
array_1 = np.ones([3,5],dtype=float)
print(array_1)

#array with single value
array_full = np.full([3,5],3.14)
print(array_full)

# Create an array filled with a linear sequence
# Starting at 0, ending at 20, stepping by 2
# (this is similar to the built-in range() function)
array_linear_sequence = np.arange(0,20,2)
print(array_linear_sequence)

# Create an array of five values evenly spaced between 0 and 1
array_linspace = np.linspace(0,1,5)
print(array_linspace)

# Create a 3x3 array of uniformly distributed
# random values between 0 and 1
array_rd = np.random.random(3)
array_rd_matrix = np.random.random((3,3))
print(array_rd)
print(array_rd_matrix)

#matrix can be indicated by [3,3] or (3,3)

# Create a 3x2 array of normally distributed random values
# with mean 0 and standard deviation 1
array_nd=np.random.normal(0,1,(3,2))
print(array_nd)

# Create a 3x1 array of random integers in the interval [0, 10)
array_ndint = np.random.randint(0, 10, (3, 1))
print(array_ndint)

# Create a 3x3 identity matrix ; that is I for AI = IA
array_identity = np.eye(3)
print(array_identity)
######################################################################################################################

# 3. basics of arrays
np.random.seed(0) # seed for reproductivity

x3 = np.random.randint(0,10,(3,4,5)) # 3-dimentional matrix
print(x3)
print(x3.ndim)
print(x3.size)
print(x3.shape)
print(x3.dtype)
print(x3.itemsize)
print(x3.nbytes)

#indexing: be mindful of 0, -1(from the end count back)

#modify a value; note the indexes used below:
x3[2,2,0] = 12
print(x3[2,2,0])

#note that type of the array is fixed; if change 12 to 12.5432, it will only show 12

#array slicing: x[start:stop:step]

x = np.arange(10)
print(x[:2])
print(x[2:])
print(x[::2]) #every other element
print(x[1::2]) #every other element from index 2; i.e. from 1

#by changing step to negative value, can reverse the array
print(x[::-1])#reverse all
print(x[5::-2]) #reversed every other from index 5

for multi-dimentional arrays
x2 = np.random.randint(0,10,(6,4))
print(x2)
print(x2[:3,:1])#1-3rd row, first column
print(x2[::2,:2]) #2,4,6row, first 2 columns
print(x2[::-1,::-1]) #reverse all

#non-copy view of array: changing a sub will also change the original
x2_sub = x2[:2,:2]
x2_sub[0,0]= 99
print(x2) #changed!

#if dont want to change:
x2_copy = x2[:2,:2].copy()
x2_copy[0,0]= 99
print(x2)#unchanged

#reshaping an array
grid = np.arange(1,10).reshape((3,3))
x = np.array([1,2,3])
print(x)
print(x.reshape((1,3)))#vector as row
print(x[np.newaxis,:])
print(x.reshape((3,1)))#vector as column
print(x[:,np.newaxis])
#np.newaxis is just repeating x itself


#concatenate arrays:
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
print(np.concatenate([x, y]))

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(np.concatenate([grid, grid]))# concatenate along the first axis
# concatenate along the second axis (zero-indexed)
print(np.concatenate([grid, grid], axis=1))

#stack
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])
# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])

#split arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
#vsplit/hsplit
grid = np.arange(16).reshape((4, 4))
print(grid)

upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

left, right = np.hsplit(grid, [2])
print(left)
print(right)

#notice np.dstack/np.dsplit: on the 3rd axis

#####################################################################################################################
4. Computation with Arrays --> numpy
np.random.seed(123)

#the first example of a loop function!!
def compute_reciprocal(values):
    output = np.empty(len(values)) #np.empty: similar to np.zeros, np.ones: Return a new array of given shape and type, without initializing entries.
    for i in range(len(values)):
        output[i] = 1.0/values[i]
    return output

values = np.arange(1,10) #if not specify 1, then the array would start from 0
print(range(len(values)))
print(values)
print(compute_reciprocal(values))

#however the loop function is actually slow; comparing to the 'vectorized operation" below:
print(1.0/values)

#generally it means arrays can be computed directly. several examples as below:
print(np.arange(5)/np.arange(1,6))

x = np.arange(9).reshape(3,3)
print(2**x) #exponential

x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division; 整除

#changing an array
print(np.add(x,2)) #add 2 to every element; same as:
print(x+2)

#some others:
print(np.subtract(x,2))
print(np.negative(x)) #negate each element
print(np.multiply(x,3))
print(np.divide(x,2))
print(np.floor_divide(x,2))#equal to x//2
print(np.power(x,3))#each element **3
print(np.remainder(x,2)) #equal to x%2

#absolute value
print(abs(np.negative(x))) #equal to:
print(np.absolute(np.negative(x)))#or
print(np.abs(np.negative(x)))

#abs also handles complex number:
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
print(np.abs(x))#output magnitude

#skipped the Trigonometric functions

#Exponential with array
x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x)) #square x on 2
print("3^x   =", np.power(3, x)) # notice that this is not x^3

x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

# Scipy.special for arrays

# Gamma functions (generalized factorials) and related functions
#Explanantion for gamma:The gamma function is often referred to as the generalized factorial since z*gamma(z) = gamma(z+1)
# and gamma(n+1) = n! for natural number n.
# beta function: beta(a,b) =  gamma(a) * gamma(b) / gamma(a+b)

x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))

# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)  =", special.erf(x)) # Returns the error function of complex argument. It is defined as 2/sqrt(pi)*integral(exp(-t**2), t=0..z).
print("erfc(x) =", special.erfc(x)) # Complementary error function, 1 - erf(x).
print("erfinv(x) =", special.erfinv(x)) # Inverse function for erfc

#simple way to store output other than creating a temporary array
#simply adding function 'OUT ='
x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

# also useful to create an array view:
y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)
# y = [0,0,0,...0], np.power(2,x) = [1,2,4,8,16]

#essentially the example is same as y[::2} = 2**x, but it actually requires to create a temporary array to store and another
#operation to copy the values into y; for large computation, it would be very inefficient

# Aggregation for Binary array
#reduce enable an operation to be applied to an array until only 1 element remains:

#sum of an array:
x = np.arange(1, 6)
print(np.add.reduce(x))

#product of all array element:
print(np.multiply.reduce(x))

#accumulate stores all the intermediate values; it returns an array
print(np.add.accumulate(x))
#check difference with np.add.reduce(x)

print(np.multiply.accumulate(x))
#check difference with np.multiply.reduce(x)

#essentially same as:
#np.sum, np.prod, np.cumsum, np.cumprod

# Outer product: self-product result
x = np.arange(1, 6)
print(np.multiply.outer(x, x)) #revise matrix calculation

3. Aggregations with Numpy (summary statistics)

#sum
L = np.random.random(100)
print(sum(L))

#sum in numpy
print(np.sum(L)) #quicker

#min & max vs np.min & np.max
print(min(L))
print(max(L))
print(np.min(L))
print(np.max(L))

print(L.min(), L.max(), L.sum())
print(np.min(L),np.max(L), np.sum(L))

#Multiple dimention aggregate
M = np.random.random((3, 4))
print(M)

#sum of all element into 1 single result
print(M.sum())

#sum by column
print(sum(M)) #compare with M.sum()
print(np.sum(M)) #same as M.sum()

print(M.sum(axis=1)) #sum by row with built-in
print(np.sum(M, axis= 1)) #sum by row with numpy

#find min
print(M.min()) #global min within M
print(M.min(axis = 0)) #min of each column of M
#same logic for max

##most aggregation has NaN-safe version, meaning that they would ignore NA values as below:
# np.sum() -> np.nansum()
# np.prod -> np.nanprod()
np.mean -> np.nanmean() #same with std, var, min, max
np.argmin -> np.nanargmin #find index of minimum #same with argmax, argmedian, percentile

#np.any (Evaluate whether any elements are true)
#np.all (Evaluate whether all elements are true)

the above 2 have no NaN safe version

######################################################################################################################
#5. Computation on Arrays: Broadcasting
import numpy as np

#non-broadcasting:
a = np.array([1,2,3])
b = np.array([5,5,5])

print(a+b)
#broadcasting:
print(a+5) #same as above but saves time

#broadcasting for higher-dimension:
M = np.ones((3,3))
M_woparenthesis = np.ones(3) #note the difference! do not forget parenthesis when coding
print(M)

print(M+a) #addition on every row

#broadcasting on both arrays:
a = np.arange(3)
b = np.arange(3)[:,np.newaxis]
print(b)
b_other = a.copy()[:, np.newaxis]
print(b_other) #same as b
b_other1 = a[:,np.newaxis]
print(b_other1) #same as b, a is not changed

#broadcasting would stretch a and b and make a new matrix:
print(a+b)

3 rules for broadcasting; but the general rule is the arrays should be able to be stretched together: size (3,3)and size (3,2) will have an error
skipped the examples

#broadcasting in practice: centering an array
X = np.random.random((10, 3))
print(X)
#calculate the mean by the row-dimension; meaning the result would be a row array of mean for each column:
Xmean = X.mean(0)
print(Xmean)

#now centering the X by mean:
X_center = X- Xmean
print(X_center)

#If we did this right, the new X_center_mean should be equal to 0 (approximately)
print(X_center.mean(0))

#notice a mistake i made here by comparing the below:
print(X_center.mean()) #without indication the dimension, the output would be only a single value

#plotting a 2-dimentional array
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
import matplotlib.pyplot as plt

plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
           cmap='viridis')
plt.colorbar()
plt.show();

#####################################################################################################################
# 6. Boolean masks with NumPy arrays

#refer to the Example 2
x = np.array([1, 2, 3, 4, 5])

#when writing  x<3, equals to Numpy: np.less(x,3)
#note: np.less(), np.greater(), np.equal() [equals to ==], np.not_equal, np.less_equal, np.greater_equal

#some examples to work with boolean value:
rng = np.random.RandomState(0) #RandomState defines the seed
x = rng.randint(10, size=(3, 4))
print(x)

print(x<6) #dtype = bool

print(np.count_nonzero(x<6)) #how many values in x are <6
print(np.sum(x<6)) #same as above; here the matrix is converted to boolean first then calculate the sum of TRUE = 1; it does not calculate on the original values of entry

#how many values < 6 on each row (axis = 1)
print(np.sum(x<6, axis = 1))

#revisit the np.all and np.any we mentioned before:
print(np.any(x>8)) #true if existing a value >8
print(np.any(x<0)) #false since the range was between 0-10, not possible to have value<0

print(np.all(x<10)) #true
print(np.all(x==6)) #false

print(np.all(x<6, axis = 1))

#boolean operators: works for string or binary digits (10 = 0000 1010; 4 = 0000 0100)
# & and : x&y = (0000 0000) 0 ; np.bitwise_and
# | or: x|y = (0000 1110) 14; np.bitwise_or
# ~ = not : ~x = (1111 0101) 245; np.bitwise_not
# ^ = XOR, x^y = (0000 1110); np.bitwise_xor, when elements equal shows false, when different shows yes
# >> right shift: x>>2 = (0000 0010) 2
#<< left shift: x<<2 = (0010 1000) 40

# In Python, all nonzero integers will evaluate as True
print(bool(42 or 0))

#when using bitwise operators (^, ~, &, | etc..), the expression operates on bits of the element
print(bin(42))

A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
print(A | B)

#use operators for boolean expression other than "and", "or'...etc, since "and" 'or' are ambiguous and will lead to error
print(A or B) #error

##############################################################################################
#6. Indexing: index mapping

rand = np.random.RandomState(111)
x = rand.randint(100, size = 10)
print(x)

#traditional indexing:
print(x[3], x[9])

#advanced indexing:
ind = [3,9]
print(x[ind])

#for multiple dimentions:
y = np.arange(12).reshape(3,4)
print(y)

row = np.array([0,1,2]) #if row = [0,1,2], it would work for pair-wise index, but wouldn't work for the multi-dimentional case below since np does not work for list objects
column = np.array([2,1,3])
print(y[row,column])

#getting a multi-dimentional output:
print(y[row[:,np.newaxis], column])

print((row[:, np.newaxis] * column).shape) #a 3*3 matrix
print(row[:, np.newaxis] * column) #as above, a 3:3 matrix; it won't work as a map for y since it is out of bound

# other combination of indexing:
print(y[2, [2, 0, 1]]) #combine with simple indexing
print(y[1:, [2, 0, 1]]) #slicing
mask = np.array([1, 0, 1, 0], dtype=bool)
print(y[row[:, np.newaxis], mask]) #row matches 'TRUE' in y; essetially column[0,2]here

#A small example: selecting random points

#first, create a random normally distributed matrix
mean = [0,0]
cov = [[1,2],
       [2,5]]
X = np.random.multivariate_normal(mean, cov,100) #refer to : http://stattrek.com/matrix-algebra/covariance-matrix.aspx to find about mean vector and covariance matrix. cov = a*a'/n, the diagonal is variance and the rest are covariance;
# always (x,x) shape for cov matrix if the original matrix is (y,x) shape
print(X.shape)

import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.scatter(X[:, 0], X[:, 1])
plt.show();

#if to choose random indices with no repeats, create a mask (fancy indexing with no repeated indices)
mapping = np.random.choice(X.shape[0],20, False) #X.shape[0] indicates how many rows; replace as false so no repetition
print(mapping)

#reminder here: mistake between() and [] could be a common bug for python! be mindful

selection = X[mapping] #fancy indexing here, no need to specify column indices!
print(selection.shape)

#over plot!
plt.scatter(X[:,0],X[:,1],alpha=0.3 )
plt.scatter(selection[:,0], selection[:, 1],facecolors ='none', edgecolors= 'blue',s=200) #need to edit edgecolors as well!
plt.show();

# useful for partitioning test/training dataset!

#modifying an array:
x = np.zeros(10)
i = np.array([2,3,1,4])
x[i]=90
print(x)

x[i]-=10
print(x)

#the following would cause unexpected result:
i = [2,3,3,4,4,4,]
x[i] +=1 #x[3] with not plus1 twice
print(x)

#in order to repeat the addition:
np.add.at(x,i,1) #repeat the addition at x[3] twice and at x[4] third times
#similar like np.negative.at(x,[2,3,4]), then 2,3,4 will turn negative
print(x)

#a small example here: binning data
np.random.seed(42)
x = np.random.randn(100) #an array of 100 elements of standard distribution (0,1); 100 here can also be multidimentional, e.g.[2,4]

bins = np.linspace(-5,5, 20) #create 20 numbers between -5 and 5
counts = np.zeros_like(bins) #Return an array of zeros with the same shape and type as a given array.

print(bins)
print(counts)

# find the appropriate bin for each x
i = np.searchsorted(bins, x)   #see below
# Explanation: -> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
# -> array([0, 5, 1, 2]), for [-10, 10, 2, 3], -10 is even before 1; 10 is more than 5, 2 is between 1-2 (the first bin)... default  'side = 'left''

print(type(i))
np.add.at(counts, i, 1)   #add 1 to each of the bins when they appear
#  very important note here: if assign counts1 = np.add.at(counts,i,1) will return nothing since it does not store results;
#  similar to when writing a loop you will use return but not print, otherwise it gives you nothing
#  simply print(counts)

print(counts)

#plot the results
import matplotlib.pyplot as plt
plt.plot(bins,counts,linestyle = 'steps')
plt.show();

plt.hist(counts,bins)
plt.show();

plt.hist(x, bins, histtype='step')
plt.show();#see the difference; we will dive deeper into visualization later
