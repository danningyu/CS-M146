# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(0, 5, 0.1);
# y = np.sin(x)
# print "hi"
# plt.plot(x, y)
# plt.show()

from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])