import numpy as np
from scipy import linalg


class LinReg:
    def __init__(self, data_path):
        #TODO construct the model

    """
    parse the csv file and construct matrix X and vector y
    @data_path: string, path to the csv file
    @return: (y, X), y is the vector of results for each training set
             X is the matrix for all training set. Each row is a training
             set.
    """
    def parse(data_path):
        #TODO

    """
    predict the value 
    @x: array-like, a value of feature vector
    @return: int, a numerical result predicted by the model
    """
    def predict(x):
        #TODO
        
    """
    plot the resulting model for the training data
    """
    def plot():
        #TODO

    """
    tranform the training data. This function can be modified case by
    case to make a better model.
    @X: np.matrix, input data
    @return: np.matrix, the data after the transformation
    """
    def __transform(X):
        #TODO
    
        
        
        
