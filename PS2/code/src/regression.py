"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

import time
import math

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        # plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        m = self.m_
        Phi = np.ones_like(X)
        for i in range(1, m+1):
            Phi = np.hstack((Phi, X**i))
        # print str(Phi)
        
        ### ========== TODO : END ========== ###
        
        return Phi
    
    
    def fit_GD(self, X, y, eta=None, eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")
        
        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        # print "Shape after poly features: " 
        # print str(X.shape)
        eta_input = eta
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        
        timeStart = time.clock()
        # GD loop
        for t in xrange(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None:
                eta = float(1)/(1+t) # change this line
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###
                
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            
            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.dot(X, self.coef_) # change this line
            # can't use self.predict() because that refits the data
            # print str(X)
            grad_res = np.dot(y_pred-y, X) #y-hat minus y, multiplied by xj
            self.coef_ = self.coef_ - 2*eta*grad_res # update J(theta)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)                
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec
        
        timeEnd = time.clock()
        print '##############################'
        print 'step size: %f' % (eta)
        print 'time spent: %f' % (timeEnd - timeStart)
        print 'number of iterations: %d' % (t+1)
        cost = 0
        for i in range(len(X)):
            cost += (y_pred.flat[i] - y[i])**2
        print 'cost: %f' % (cost)
        print 'coefficients: '
        for i, coef in zip(range(len(self.coef_)), self.coef_):
            print str(i) + ': ' +str(coef) 
        # for (coef, i) in self.coef_, range(len(self.coef_)):
        #     print str(i) + ': ' +str(coef)
        return self
    
    
    def fit(self, X, y, l2regularize = None ) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
        """
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        timeStart = time.clock()
        self.coef_ = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)
        timeEnd = time.clock()
        print 'Time spent for exact solution: %f' %(timeEnd - timeStart)

        # print self.coef_
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        y = 0
        y = np.dot(X, self.coef_) # (n,d) * (d,) = (n,)
        ### ========== TODO : END ========== ###
        # print 'Predicted: '
        # print(y)
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)
        y_predicted = self.predict(X)
        n,d = X.shape
        cost = 0
        for i in range(len(X)):
            cost += (y_predicted.flat[i] - y[i])**2
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        error = 0
        error = math.sqrt(self.cost(X, y)/len(y))
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        # plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
  
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    print 'Visualizing data...'
    # plot_data(train_data.X, train_data.y)
    # plot_data(test_data.X, test_data.y)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression
    print 'Investigating linear regression...'
    polyreg = PolynomialRegression()
    polyreg.generate_polynomial_features(train_data.X)
    polyreg.generate_polynomial_features(test_data.X)

    # Test code, should output 40.234
    # train_data = load_data('regression_train.csv')
    # model = PolynomialRegression()
    # model.coef_ = np.zeros(2)
    # cost = model.cost(train_data.X, train_data.y)
    # print 'Cost: ' + str(cost)
    # print str(train_data.X)
    # print str(train_data.X.shape)
    # print str(train_data.y)

    eta_vals = [0.0001, 0.001, 0.01, 0.0407]
    for eta_val in eta_vals:
        polyreg.fit_GD(train_data.X, train_data.y, eta = eta_val, verbose=False)
    # polyreg.fit_GD(train_data.X, train_data.y)
    polyreg.fit(train_data.X, train_data.y)
    print 'Cost: %f' %(polyreg.cost(train_data.X, train_data.y))
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    print 'Investigating polynomial regression...'
    train_errors = []
    test_errors = []
    min_test_error = 1000
    min_error_index = -1
    for i in range(11):
        polyreg_higher = PolynomialRegression(i)
        polyreg_higher.fit(train_data.X, train_data.y)
        train_error = polyreg_higher.rms_error(train_data.X, train_data.y)
        test_error = polyreg_higher.rms_error(test_data.X, test_data.y)
        train_errors.append(train_error)
        test_errors.append(test_error)
        if test_error < min_test_error:
            min_test_error = test_error
            min_error_index = i
    complexities = range(11)
    plt.plot(complexities, train_errors, label = "Training RMSE")
    plt.plot(complexities, test_errors, label ="Test RMSE")
    plt.xlabel("Model Complexity: Polynomial Degree")
    plt.ylabel("Root Mean Square Error (RSME)")
    plt.legend()
    plt.title("Problem 4(i): Polynomial Regression")
    plt.show()
    print 'Min test error: %f' %(min_test_error)
    print 'At degree %d' %(min_error_index)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts j-k (extra credit): main code for regularized regression
    print 'Investigating regularized regression...'
        
    ### ========== TODO : END ========== ###
    
    
    
    print "Done!"

if __name__ == "__main__" :
    main()
