"""
Author      : Yi-Chieh Wu, Sriram Sankararman
Description : Twitter
"""

from string import punctuation

import numpy as np

# !!! MAKE SURE TO USE SVC.decision_function(X), NOT SVC.predict(X) !!!
# (this makes ``continuous-valued'' predictions)
from sklearn.svm import SVC
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile):
    """
    Writes your label vector to the given file.
    
    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """
    
    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return
    
    np.savetxt(outfile, vec)    


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string):
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.
    
    Parameters
    --------------------
        input_string -- string of characters
    
    Returns
    --------------------
        words        -- list of lowercase "words"
    """
    
    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile):
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.
    
    Parameters
    --------------------
        infile    -- string, filename
    
    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    
    word_list = {}
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1a: process each line to populate word_list
        word_counter = 0
        for line in fid:
            line_split = extract_words(line)
            for word in line_split:
                if word not in word_list:
                    word_list[word] = word_counter
                    word_counter += 1

        ### ========== TODO : END ========== ###
    # print str(word_list)
    return word_list


def extract_feature_vectors(infile, word_list):
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.
    
    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)
    
    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """
    
    num_lines = sum(1 for line in open(infile,'rU'))
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))
    
    with open(infile, 'rU') as fid :
        ### ========== TODO : START ========== ###
        # part 1b: process each line to populate feature_matrix
        tweet_number = 0
        for tweet in fid:
            words = extract_words(tweet)
            for word in words:
                feature_matrix[tweet_number][word_list[word]] = 1
            tweet_number += 1
        ### ========== TODO : END ========== ###
    # print str(feature_matrix)
    return feature_matrix


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1-score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_label)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)
    
    confusion_matrix = metrics.confusion_matrix(y_true, y_label)
    tn, fp, fn, tp = confusion_matrix.ravel()
    if metric == "sensitivity":
        return float(tp)/float(tp+fn)
    elif metric == "specificity":
        return float(tn)/float(tn+fp)
    
    return -1 #bad argument for metric
    ### ========== TODO : END ========== ###


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    ### ========== TODO : START ========== ###
    # part 2b: compute average cross-validation performance    
    totalError = 0
   #  print str(kf)
    totalTests = 0
    for train_index, test_index in kf.split(X, y):
        # print str(train_index) + ',' + str(test_index)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.decision_function(X_test)
        totalError += performance(y_test, y_pred, metric)
        totalTests += 1

    # return average error
    return float(totalError)/totalTests
    ### ========== TODO : END ========== ###


def select_param_linear(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    best_performance = -1
    best_C = -1
    for C_value in C_range:
        clfLinear = SVC(C=C_value, kernel="linear")
        score = cv_performance(clfLinear, X, y, kf, metric)
        print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ': C=' + str(C_value) + ': %.4f' % score
        if score >= best_performance:
            best_C = C_value
            best_performance = score
    print 'Best C:' + str(best_C)
    return best_C
    ### ========== TODO : END ========== ###


def select_param_rbf(X, y, kf, metric="accuracy"):
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- cross_validation.KFold or cross_validation.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        gamma, C -- tuple of floats, optimal parameter values for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-5, 3)
    best_C = -1
    best_gamma = -1
    best_performance = -1
    for C_val in C_range:
        for gamma_val in gamma_range:
            clfRBF = SVC(C=C_val, kernel="rbf", gamma=gamma_val)
            score = cv_performance(clfRBF, X, y, kf, metric)
            print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ': C=' + str(C_val) + ': gamma=' + str(gamma_val) + ': %.4f' % score
        if score > best_performance:
            best_C = C_val
            best_gamma = gamma_val
            best_performance = score

    return best_C, best_gamma
    ### ========== TODO : END ========== ###


def performance_test(clf, X, y, metric="accuracy"):
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
    """

    ### ========== TODO : START ========== ###
    # part 4b: return performance on test data by first computing predictions and then calling performance
    y_pred = clf.decision_function(X)
    score = performance(y, y_pred, metric)   
    return score
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################
 
def main() :
    np.random.seed(1234)
    
    # read the tweets and its labels   
    dictionary = extract_dictionary('../data/tweets.txt')
    X = extract_feature_vectors('../data/tweets.txt', dictionary)
    y = read_vector_file('../data/labels.txt')
    
    metric_list = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    
    ### ========== TODO : START ========== ###
    # part 1c: split data into training (training + cross-validation) and testing set
    training_X = X[:560]
    training_y = y[:560]
    test_X = X[560:]

    test_y = y[560:]
    # print str(test_X)
    # print str(test_y)
    # part 2b: create stratified folds (5-fold CV)
    skf = StratifiedKFold(n_splits=5)

    # part 2d: for each metric, select optimal hyperparameter for linear-kernel SVM using CV
    optimal_C_values = {}
    # for metric in metric_list:
    #     result = select_param_linear(training_X, training_y, skf, metric)
    #     optimal_C_values[metric] = result
    # print str(optimal_C_values)

    # part 3c: for each metric, select optimal hyperparameter for RBF-SVM using CV
    # optimal_C_values.clear()
    # optimal_gamma_values = {}
    # for metric in metric_list:
    #     best_C, best_gam = select_param_rbf(training_X, training_y, skf, metric)
    #     optimal_C_values[metric] = best_C
    #     optimal_gamma_values[metric] = best_gam
    # print str(optimal_C_values)
    # print str(optimal_gamma_values)
    
    # part 4a: train linear- and RBF-kernel SVMs with selected hyperparameters
    C_val = 100.0
    gamma_val = 0.01
    clfLinear = SVC(C=C_val, kernel="linear")
    clfRBF = SVC(C=C_val, kernel="rbf", gamma=gamma_val)
    clfLinear.fit(training_X, training_y)
    clfRBF.fit(training_X, training_y)

    # part 4c: report performance on test data
    for metric in metric_list:
        result = performance_test(clfLinear, test_X, test_y, metric)
        print 'Linear SVM, metric: ' + str(metric) + ' score: %.4f' %result
        result = performance_test(clfRBF, test_X, test_y, metric)
        print 'RBF SVM, metric: ' + str(metric) + ' score: %.4f' %result

    ### ========== TODO : END ========== ###
    
    print 'Done!'
    
if __name__ == "__main__" :
    main()
