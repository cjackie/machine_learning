import numpy as np
from util import parse

class SVM:
    
    """
    assume data format is correct and y is either 1 or -1
    @data_path, string, path to data
    @C, float, panelty weight
    @epsilon, float, accepted converging error
    @show_prog, boolean, indicate if needed to print the progress
    """
    def __init__(self, data_path, C, epsilon, show_prog):
        _, y, X = parse(data_path)
        self._show_prog = show_prog
        b, alpha = self.__optimize(y, X, C, epsilon)
        alpha_i = []
        for i in range(len(alpha)):
            if alpha.item(i,0) != 0:
                alpha_i.append(i)

        self._b = b
        self._y = y
        self._X = X
        self._alpha = alpha
        self._alpha_i = alpha_i


    """
    predict given an input vector
    @x, np.matrix, a vector
    @return, int, -1 or 1
    """
    def predict(self, x):
        X = self._X
        y = self._y
        b = self._b
        a = self._alpha
        a_i = self._alpha_i
        K = self.__K
        m = np.matrix

        u = 0.0
        for i in a_i:
            u = u + y.item(i,0)*a.item(i,0)*K(m(X.A[i]).T,x)
        return 1 if u>0 else -1

    """
    the method to find the optimal array of alpha
    @y, np.matrix, a vector
    @X, np.matrix, matrix of training data. m x n, where m is number
                   of training example, and n is input features
    @C, float, panelty weight
    @epsilon, float, accepted converging error
    @return, (int, np.matrix), int is intercept term, and np.matrix is 
                               alpha
    """
    def __optimize(self, y, X, C, epsilon):
        #TODO

    """
    optimize the function using coordinate descend on 2 variables
    @i1, int, index to alpha array
    @i2, int, index to alpha array
    @return, (float, float) or None, the updated vaule for elems in i1 and i2
             None means i1 and i2 are bad pair to optimize, change them.
    """
    def __step(self, i1, i2):
        m = np.matrix
        u = self.__u
        K = self.__K
        a1, a2 = self._alpha.item(i1,0), self._alpha.item(i2,0)
        x1, x2 = m(self._X.A[i1]).T, m(self._X.A[i2]).T
        y1, y2 = self._y.item(i1,0), self._y.item(i2,0)
        L = max(0, a1+a2-self._C) if y1==y2 else max(0, a2-a1)
        H = min(self._C, a2+a1) if y1==y2 else min(self._C, self._C+a2-a1)

        if L==H or i1==i2:
            return None
            
        eta = K(x1,x1)+K(x2,x2)-2*K(x1,x2)
        E1 = u(x1) - y1
        E2 = u(x2) - y2

        a2_n = a2+y2*(E1-E2)/eta
        if a2_n < L:
            a2_nc = L
        elif a2_n > H:
            a2_nc = H
        else:
            a2_nc = a2_n
        a1_n = a1+y1*y2*(a2-a2_nc)
        return a1_n, a2_nc


    """
    the kernel, which can be modified on case-by-case basis
    @x1, np.matrix, a vector
    @x2, np.matrix, a vector
    @return, float, the result of the kernel
    """
    def __K(self, x1, x2):
        #customizable
        return x1.T*x2.item(0,0) #no transformation

    """
    support vector machine function, it will get parameters via 
    field variable(program it with side effects).
    @x, np.matrix, a vector
    @return, float, the result
    """
    def __u(self, x):
        b = self._b
        K = self._K
        X = self._X
        y = self._y
        alpha = self._alpha
        m = np.matrix

        u = 0.0
        for i in range(len(X)):
            u = u + y.item(i,0)*alpha.item(i,0)*K(m(X.A[i]).T,x)
        return u-b
        
