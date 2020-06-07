import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from numpy import linalg as LA

# Practice with plotting
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# plt.ylabel('some numbers')
# plt.xlabel('x-axis yay')
# plt.axis([0, 6, 0, 20]) #xmin, xmax, ymin, ymax
# plt.show()

# t = np.arange(0, 5, 0.2)
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()

# data = {'a': np.arange(50),
# 		'c': np.random.randn(0, 50, 50),
# 		'd': np.random.randn(50)}
# data['b'] = data['a']+10*np.random.randn(50)
# data['d'] = np.abs(data['d'])*100

# plt.scatter('a', 'b', c='c', s='d', data = data)
# plt.show()

# ps0 Problem 10(a)
# x_list = []
# y_list = []
# for i in range(1000):
# 	coord = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]])
# 	# first array is mean, second is covariance
# 	x_list.append(coord[0])
# 	y_list.append(coord[1])
# plt.plot(x_list, y_list, 'o', markersize=5)
# plt.xlabel(r'$\ x_1$')
# plt.ylabel(r'$\ y_1$')
# plt.title('10(a): 1000 Points From a 2D Gaussian Distribution')
# plt.show()

# Problem 10(b)
# x_list2 = []
# y_list2 = []
# for i in range(1000):
# 	coord = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]])
# 	x_list2.append(coord[0])
# 	y_list2.append(coord[1])
# plt.plot(x_list2, y_list2, 'o', markersize=5)
# plt.xlabel(r'$\ x_1$')
# plt.ylabel(r'$\ y_1$')
# plt.title('10(b): 1000 Points From a 2D Gaussian Distribution Centered at (1, 1)')
# plt.show()

# Problem 10(c): double the covariance
# x_list = []
# y_list = []
# for i in range(1000):
# 	coord = np.random.multivariate_normal([0, 0], [[2, 0], [0, 2]])
# 	# first array is mean, second is covariance
# 	x_list.append(coord[0])
# 	y_list.append(coord[1])
# plt.plot(x_list, y_list, 'o', markersize=5)
# plt.xlabel(r'$\ x_1$')
# plt.ylabel(r'$\ y_1$')
# plt.title('10(c): 1000 Points From a 2D Gaussian Distribution')
# plt.show()

# Problem 10(d): change covariance
# x_list = []
# y_list = []
# for i in range(1000):
# 	coord = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
# 	# first array is mean, second is covariance
# 	x_list.append(coord[0])
# 	y_list.append(coord[1])
# plt.plot(x_list, y_list, 'o', markersize=5)
# plt.xlabel(r'$\ x_1$')
# plt.ylabel(r'$\ y_1$')
# plt.title('10(d): 1000 Points From a 2D Gaussian Distribution')
# plt.show()

# Problem 10(e): change covariance
# x_list = []
# y_list = []
# for i in range(1000):
# 	coord = np.random.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]])
# 	# first array is mean, second is covariance
# 	x_list.append(coord[0])
# 	y_list.append(coord[1])
# plt.plot(x_list, y_list, 'o', markersize=5)
# plt.xlabel(r'$\ x_1$')
# plt.ylabel(r'$\ y_1$')
# plt.title('10(e): 1000 Points From a 2D Gaussian Distribution')
# plt.show()

# Problem 11: calculate eigenvectors
a = np.array([[1, 0], [1, 3]])
evalues, evectors = LA.eig(a)
# print(evalues)
# print(evectors)
index_of_max = np.where(evalues==evalues.max())
print(evectors[:,index_of_max].reshape(2, 1)) #reshape to "flatten" it