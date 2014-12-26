import numpy as np
import matplotlib.pyplot as plt

class LinReg:
    """
    self.names: an array of names for each feature
    self.theta: np.matrix of linear coefficients, a vector
    """
    def __init__(self, data_path):
        names, y, X = self.parse(data_path)
        X = self.__transform(X)
        self.names = names
        self.theta = (X.T*X).I*X.T*y
        self.__plot(X, y)

    """
    parse the csv file and construct matrix X and vector y
    @data_path: string, path to the csv file
    @return: (names ,y, X), y is the vector of results for each training set
             X is the matrix for all training set. Each row is a training
             set. names is an array of label for data
    """
    def parse(self,data_path):
        f = open(data_path)
        line = f.readline()
        names = line[:-1].split(",")
        X, y = [], []
        for line in f:
            row = [int(a) for a in line.split(",")]
            y.append([row.pop(0)])
            X.append([1]+row)                        #include the intercept term
        return names, np.matrix(y), np.matrix(X)

    """
    predict the value 
    @x: 1d-array-like, a value of feature vector
    @return: int, a numerical result predicted by the model
    """
    def predict(self,x):
        x = [1] + x
        x_vector = self.__transform_v(np.matrix(x).T)
        if len(x) != len(self.theta):
            print("data len is inconsistent")
            return
        return (b*b.T).tolist()[0][0]
        
    """
    plot the resulting model for the training data
    @y: the vector of results for each training set
    @X: the matrix for all training set. Each row is a training
        set.
    """
    def __plot(self,X,y):
        #TODO dimension is at most 3
        
    """
    tranform the training data. This function can be modified case by
    case to make a better model.
    @X: np.matrix, input data
    @return: np.matrix, the data after the transformation
    """
    def __transform(self,X):
        #customizable
        return [self.transfrom_v(x) for x in X]

    """
    tranform a input vector.
    @x: np.matrix, a row, 1*n dimension
    @return: np.matrix, the data after the transformation
    """
    def __transform_v(self,x):
        #customizable
        return x
    
        
        
        
