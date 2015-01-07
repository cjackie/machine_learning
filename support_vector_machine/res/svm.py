import numpy as np
from random import random
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
        X,y,b,a,a_i,K,m = self._X,self._y,self._b,self._alpha,self._alpha_i,self.__K
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
    @return, (float, np.matrix), float is intercept term, and np.matrix is 
                                 alpha
    """
    def __optimize(self, y, X, C, epsilon):
        m = np.matrix
        self._y = y
        self._X = X
        self._C = C
        self._b = 0
        self._epsilon = epsilon
        self._alpha = m([C/2.0]*len(y)).T

        #check if given alpha and index to the alpha
        #return true is condition is meet, false otherwsie
        def check(a, a_i):
            e = epsilon
            u = self.__u
            if a.item(a_i,0) == 0:
                return y.item(a_i,0)*u(np.matrix(X.A[a_i]).T) >= 1-e
            elif a.item(a_i,0) > 0 and a.item(a_i,0) < C:
                return y.item(a_i,0)*u(np.matrix(X.A[a_i]).T) >= 1-e \
                    and y.item(a_i,0)*u(np.matrix(X.A[a_i]).T) <= 1+e 
            elif a.item(a_i,0) == C:
                return y.item(a_i,0)*u(np.matrix(X.A[a_i]).T) <= 1+e
            else:
                return False

        #return array of index of alphas that don't meet the condition
        def check_cond(alpha):
            a_i = []
            for i in range(len(alpha)):
                if not check(alpha, i):
                    a_i.append(i)
            return a_i

        #main loop for the optimization
        while True:
            a = self._alpha
            update = self.__update
            
            a_i = check_cond(a)
            for i in a_i:
                update(i)
                
            a_i = check_cond(a)
            a_i_nonb = []
            for i in a_i:
                if a.item(i,0) > 0 and a.item(i,0) < C:
                    a_i_non_b.append(i)

            while True:
                for i in a_i_nonb:
                    update(i)
                    
                finished = True
                for i in a_i_nonb:
                    if not check(a, i):
                        finished = False
                        break
                    
                if finished:
                    break

            a_i = check_cond(a)
            if len(a_i) == 0:
                break

        return self._b, self._alpha
                
                    
    """
    optimized with respect to index to i2 in the vector alpha and b, it will find
    the second alpha to update.
    @i2, int, index to alpha array
    """
    def __update(self, i2):
        a,b,s,K,y,X,C,u = self._alpha,self._b,self.__step,self.__K, \
                        self._y,self._X,self._C,self.__u
        m = np.matrix

        i1 = random()*len(a)
        i1 = i1+1 if i1 == i2 else i1
        a1, a2 = s(i1, i2)
        a1_o,a2_o = a.A[i1][0],a.A[i2][0]
        y1,y2,x1,x2 = y.item(i1,0),y.item(i2,0),m(X.A[i1]).T,m(X.A[i2]).T

        b1 = (u(x1)-y1) + y1*(a1-a1_o)*K(x1,x1)+y2*(a2-a2_o)*K(x1,x2)+b
        b2 = (u(x2)-y2) + y1*(a1-a1_o)*K(x1,x2)+y2*(a2-a2_o)*K(x2,x2)+b
        if not (a1>0 and a1<C):
            b = b1
        elif not (a2>0 and a2<C):
            b=b2
        else:
            b = (b1+b2)/2
        
        a.A[i1][0] = a1
        a.A[i2][0] = a2
        self._b = b

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
        b,X,y,alpha,K,m = self._b,self._X,self._y,self._alpha,self.__K,np.matrix

        u = 0.0
        for i in range(len(X)):
            u = u + y.item(i,0)*alpha.item(i,0)*K(m(X.A[i]).T,x)
        return u-b
        
