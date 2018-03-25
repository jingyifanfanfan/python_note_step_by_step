#Refer to the readme file before use

#1. Data type in Python

L = list(range(10))
print(L)
print(type(L))
print(type(L[1])) #check the difference

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

# #2. create an array
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

#for multi-dimentional arrays
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

#3. Computation with Arrays --> numpy
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
