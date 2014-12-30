import numpy as np
import scipy.linalg as linalg

class LogReg:
    
    STOCHASTIC_GRADIENT_ASCENT = 'stoch'
    GRADIENT_ASCENT = 'grad'
    NEWTON_METHOD = 'newton'
    
    def __init__(self, data_path, asc_deg=0.1,
                 opt_code=STOCHASTIC_GRADIENT_ASCENT, show_prog=False):
        #TODO contruct the model

    """
    parse the data and return result vector and the data matrix, assume 
    data format is csv with first column being output
    @data_path: string, the absolute data path to the data
    @return, (y, X), both type np.matrix, y is a vector, X is matrix with 
             row being each data, and column being features
    """
    def parse(self, data_path):
        #TODO

    """
    predict the result by the model
    @x, np.matrix, a vector
    @return, float, the result
    """
    def predict(self, x):
        #TODO
        
    """
    choose the corresponding method accorrding to opt_code
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @opt_code, string, to indicate which optimization algorithm to run
    @alpha, float, "step size" for optimization algorithm
    @return, np.matrix, a vector for parameter theta
    """
    def __compute_parameters(self, y, X, opt_code, alpha):
        #TODO

    """
    stochastic gradient ascent
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm
    @return, np.matrix, a vector for parameter theta
    """
    def __stoch(self, y, X, alpha):
        #TODO
        
    """
    gradient ascent
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm
    @return, np.matrix, a vector for parameter theta
    """
    def __grad(self, y, X, alpha):
        #TODO
        
    """
    newton's method
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm
    @return, np.matrix, a vector for parameter theta
    """
    def __newton(self, y, X, alpha):
        #TODO
        
        
        
