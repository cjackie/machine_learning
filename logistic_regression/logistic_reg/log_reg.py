import numpy as np
import scipy.linalg as linalg
from numpy.linalg import norm
from util import parse
from time import clock

class LogReg:
    
    STOCHASTIC_GRADIENT_ASCENT = 'stoch'
    GRADIENT_ASCENT = 'grad'
    NEWTON_METHOD = 'newton'
    
    def __init__(self, data_path, resolution=0.1**4, alpha=0.1, \
                 opt_code='stoch', show_prog=False):
        (names, y, X) = parse(data_path)
        self._show_prog = show_prog
        self._theta = self.__compute_parameters(y, X, opt_code, alpha, resolution)
        
    """
    predict the result by the model
    @x, np.matrix, a vector
    @return, float, the result
    """
    def predict(self, x):
        e = np.e
        return 1.0/(1+e**(-(self._theta.T*x).item(0,0)))
        
    """
    choose the corresponding method accorrding to opt_code
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @opt_code, string, to indicate which optimization algorithm to run
    @alpha, float, "step size" for optimization algorithm
    @resolution, float, accepted converging number
    @return, np.matrix, a vector for parameter theta
    """
    def __compute_parameters(self, y, X, opt_code, alpha, resolution):
        if opt_code == self.STOCHASTIC_GRADIENT_ASCENT:
            return self.__stoch(y, X, alpha, resolution)
        elif opt_code == self.GRADIENT_ASCENT:
            return self.__grad(y, X, alpha, resolution)
        elif opt_code == self.NEWTON_METHOD:
            return self.__newton(y, X, alpha, resolution)
        else:
            print("invalid opt_code!!, exiting")
            exit(-1)

    """
    stochastic gradient ascent
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm    
    @resolution, float, accepted converging number
    @return, np.matrix, a vector for parameter theta
    """
    def __stoch(self, y, X, alpha, resolution):
        theta = np.matrix([0]*len(X.T)).T
        h = lambda x, theta: 1.0/(1+np.e**(-(theta.T*x).item(0,0)))    #model
        if self._show_prog:
            before = clock()
            iter_c = 1
        for i in range(len(X.A)):
            x = np.matrix(X.A[i]).T
            delta = alpha*(y.A[i][0] - h(x, theta))*x
            theta = theta + delta
            if self._show_prog:
                print(("for row %d: %s\n" + \
                       "increase by: %s\n" + \
                       "theta after updating: %s") \
                      % (iter_c,X.A[i],delta.flatten().A[0],theta.flatten().A[0]));
                iter_c += 1
        if self._show_prog:
            print("time spending on optimization: %f", clock()-before)
        return theta
        
    """
    gradient ascent
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm
    @resolution, float, accepted converging number
    @return, np.matrix, a vector for parameter theta
    """
    def __grad(self, y, X, alpha, resolution):
        #return a function to compute gradient vector
        def gradient(y,X):
            def g(theta):
                m = np.matrix
                h = lambda x, theta: 1.0/(1+np.e**(-(theta.T*x).item(0,0)))    #model
                h_v = m([h(m(x).T, theta) for x in X.A]).T
                return X.T*(y-h_v)
            return g
        
        g = gradient(y,X)
        theta = np.matrix([0]*len(X.T)).T
        if self._show_prog:
            before = clock()
        while True:
            g_v = g(theta)
            theta = theta + alpha*g_v
            if self._show_prog:
                print("gradient: %s, theta is: %s" % (g_v.flatten().A[0], theta.flatten().A[0]))
            if (norm(g_v) < resolution):
                if self._show_prog:
                    print("the time spent on optimization: %f" % (clock()-before))
                return theta
        
    """
    newton's method
    @y, np.matrix, a vector
    @X, np.matrix, X is matrix with row being each data, and column being features
    @alpha, float, "step size" for optimization algorithm
    @resolution, float, accepted converging number
    @return, np.matrix, a vector for parameter theta
    """
    def __newton(self, y, X, alpha, resolution):
        def gradient(y,X):
            def g(theta):
                m = np.matrix
                h = lambda x, theta: 1.0/(1+np.e**(-(theta.T*x).item(0,0)))    #model
                h_v = m([h(m(x).T, theta) for x in X.A]).T
                return X.T*(y-h_v)
            return g
        
        def hessian(y,X):
            def h(theta):
                dim = len(X.T)
                H = [[0 for i in range(dim)]
                        for j in range(dim)]
                e = np.e
                m = np.matrix
                for i in range(dim):
                    for j in range(dim):
                        sum = 0
                        for x in X.A:
                            sum += x[i]*x[j]*e**(-theta.T*m(x).T) \
                                   /((1+e**((-theta.T*m(x).T).A[0][0]))**2)
                        H[i][j] = sum
                return m(H)
            return h

        g = gradient(y,X)
        h = hessian(y,X)
        theta = np.matrix([0]*len(X.T)).T
        if self._show_prog:
            before = clock()
        while True:
            if norm(g(theta)) < resolution:
                if self._show_prog:
                    print("total time spent on optimization: %f", clock()-before)
                return theta
            theta = theta - h(theta).I*g(theta)
            if self._show_prog:
                print("gradient: %s, theta: %s" % (g(theta).T.A, theta.T.A))

    """
    immutable getter to theta
    """
    @property
    def theta(self):
        return np.matrix(self._theta)
        
