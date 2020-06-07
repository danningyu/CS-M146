"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        Derived class of Classfier superclass.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n
        # array of self.prediction_, n times
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        totalPoints = y.shape[0]
        numSurvived = 0
        numDied = 0
        for survivalValue in y:
            if survivalValue == 0:
                numDied += 1
            elif survivalValue == 1:
                numSurvived += 1
        self.probabilities_ = {
            0: float(numDied)/totalPoints, 
            1: float(numSurvived)/totalPoints
        }
       # self.probabilities_[0] = 
        # self.probabilities_[1] = 
       #  print "proportional of deaths: " + str(self.probabilities_[0])
        # print "proportional of survivals: " + str(self.probabilities_[1])
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        outcomes = [0, 1]
        y = np.random.choice(outcomes, X.shape[0], [self.probabilities_[0], self.probabilities_[1]])
        # outcomes are 0 or 1, X.shape[0] gives num of elements to generate
        # last arg gives probability for 0 and probability for 1
        ### ========== TODO : END ========== ###
        
        return y

######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in xrange(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0 
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred_from_train = clf.predict(X_train)
        y_pred_from_test = clf.predict(X_test)
        train_error += float(1-metrics.accuracy_score(y_train, y_pred_from_train, normalize=True))
        test_error += float(1-metrics.accuracy_score(y_test, y_pred_from_test, normalize=True))

    train_error /= float(ntrials)
    test_error /= float(ntrials)
        
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():

    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print 'Plotting...'
    for i in xrange(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show=True)
        #change to show=True to get the histograms

       
    #========================================
    # train Majority Vote classifier on data
    print 'Classifying using Majority Vote...'
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print '\t-- training error: %.3f' % train_error
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print 'Classifying using Random...'
    clf2 = RandomClassifier()
    clf2.fit(X, y)
    y_pred2 = clf2.predict(X)
    train_error2 = 1-metrics.accuracy_score(y, y_pred2, normalize=True)
    print '\t-- training error: %.3f' % train_error2
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print 'Classifying using Decision Tree...'
    clf3 = DecisionTreeClassifier(criterion="entropy")
    clf3.fit(X, y)
    y_pred3 = clf3.predict(X)
    train_error3 = 1-metrics.accuracy_score(y, y_pred3, normalize=True)
    print '\t-- training error: %.3f' % train_error3
    
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    
    
    ### ========== TODO : START ========== ###
    # part d: use cross-validation to compute average training and test error of classifiers
    print 'Investigating various classifiers...'
    clfMajorityCV = MajorityVoteClassifier()
    clfRandomCV = RandomClassifier()
    clfDecisionTreeCV = DecisionTreeClassifier(criterion="entropy")
    majority_train_error, majority_test_error = error(clfMajorityCV, X, y)
    random_train_error, random_test_error = error(clfRandomCV, X, y)
    tree_train_error, tree_test_error = error(clfDecisionTreeCV, X, y)
    print '\t-- majority training error: %.3f' % majority_train_error
    print '\t-- majority testing error: %.3f' % majority_test_error
    print '\t-- random training error: %.3f' % random_train_error
    print '\t-- random testing error: %.3f' % random_test_error
    print '\t-- decision training error: %.3f' % tree_train_error
    print '\t-- decision testing error: %.3f' % tree_test_error
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: investigate decision tree classifier with various depths
    print 'Investigating depths...'
    min_depth = 1
    max_depth = 20
    tree_limit_depth_train_error = []
    tree_limit_depth_test_error = []
    random_cv_error_baseline = []
    majority_cv_error_baseline = []
    trash_list = []
    depths = range(min_depth, max_depth+1) #the range [1, 21)
    for i in depths:
        # range(a, b) is the range [a, b)
        clf_tree_limit_depth = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        clf_random_cv_baseline = RandomClassifier()
        clf_majority_cv_baseline = MajorityVoteClassifier()
        limit_depth_train_error, limit_depth_test_error = error(clf_tree_limit_depth, X, y)
        tree_limit_depth_train_error.append(limit_depth_train_error)
        tree_limit_depth_test_error.append(limit_depth_test_error);
        trash_list, random_error = error(clf_random_cv_baseline, X, y)
        random_cv_error_baseline.append(random_error)
        trash_list, maj_error = error(clf_majority_cv_baseline, X, y)
        majority_cv_error_baseline.append(maj_error)
    plt.plot(depths, tree_limit_depth_train_error, label = "Decision Tree Training Error")
    plt.plot(depths, tree_limit_depth_test_error, label = "Decision Tree Test Error")
    plt.plot(depths, random_cv_error_baseline, label = "Random Classifier Test Error")
    plt.plot(depths, majority_cv_error_baseline, label = "Majority Vote Classifier Test Error")
    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Train or Test Error")
    plt.xticks(np.arange(0, max_depth+1, step=2))
    plt.legend()

    i = 1
    min_error = 1
    best_depth = -1
    for error_val in tree_limit_depth_test_error:
        if error_val <= min_error:
            best_depth = i
            min_error = error_val
        i+=1
    print 'Best error is {0:.3f} at depth {1}'.format(min_error, best_depth)


    plt.show()

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part f: investigate decision tree classifier with various training set sizes
    print 'Investigating training set sizes...'
    tree_train_size_train_errors = []
    tree_train_size_test_errors = []
    random_train_size_test_errors = []
    majority_train_size_test_errors = []
    training_splits = np.arange(0.05, 0.95+0.05, step=0.05)
    for split_amt in training_splits:
        clf_tree_diff_train_sizes = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth)
        clf_random_ts_base = RandomClassifier()
        clf_maj_ts_base = MajorityVoteClassifier()
        train_error_train_size, test_error_train_size = error(clf_tree_diff_train_sizes, X, y, test_size=1-split_amt)
        tree_train_size_train_errors.append(train_error_train_size)
        tree_train_size_test_errors.append(test_error_train_size)
        trash_list, random_ts_test_error = error(clf_random_ts_base, X, y, test_size=1-split_amt)
        random_train_size_test_errors.append(random_ts_test_error)
        trash_list, majority_ts_test_error = error(clf_maj_ts_base, X, y, test_size=1-split_amt)
        majority_train_size_test_errors.append(majority_ts_test_error)

    plt.plot(training_splits, tree_train_size_train_errors, label="Decision Tree Training Error")
    plt.plot(training_splits, tree_train_size_test_errors, label="Decision Tree Test Error")
    plt.plot(training_splits, random_train_size_test_errors, label="Random Classifier Test Error")
    plt.plot(training_splits, majority_train_size_test_errors, label="Majority Vote Classifier Test Error")
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.legend()
    plt.xlabel("Proportion of Training Data")
    plt.ylabel("Train or Test Error")
    plt.show()
    ### ========== TODO : END ========== ###
    
       
    print 'Done'


if __name__ == "__main__":
    main()
