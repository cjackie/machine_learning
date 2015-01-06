import numpy as np
from util import parse

class SVM:

    def __init__(self, data_path, C, epsilon, show_prog):
        #TODO build the model

    """
    predict given an input vector
    @x, np.matrix, a vector
    @return, int, 0 or 1(in svm 0 means -1 and 1 means 1)
    """
    def predict(self, x):
        #TODO

    """
    the method to find the optimal array of alpha
    @y, np.matrix, a vector
    @X, np.matrix, matrix of training data. m x n, where m is number
                   of training example, and n is input features
    @return, (int, np.matrix), int is intercept term, and np.matrix is 
                               alpha
    """
    def __optimize(self, y, X):
        #TODO

    """
    optimize the function using coordinate descend on 2 variables
    @i1, int, index to alpha array
    @i2, int, index to alpha array
    """
    def __step(self, i1, i2):
        #TODO

    """
    the kernel, which can be modified on case-by-case basis
    @x1, np.matrix, a vector
    @x2, np.matrix, a vector
    @return, float, the result of the kernel
    """
    def __K(self, x1, x2):
        #customizable
        #TODO

        

    
