import numpy as np
from random import random
from util import parse
from time import clock
import matplotlib.pyplot as plt

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
        X,y,b,a,a_i,K = self._X,self._y,self._b,self._alpha,self._alpha_i,self.__K
        m = np.matrix

        u = 0.0
        for i in a_i:
            u = u + y.item(i,0)*a.item(i,0)*K(x,m(X.A[i]).T)
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
        self._E = m([0]*len(y)).T
        if self._show_prog and len(X.T) == 2:
            self._before = clock()
            plt.ion()
            x1,y1,x2,y2 = [],[],[],[]
            for i in range(len(X)):
                if y.item(i,0) == -1:
                    x1 = x1 + [X.item(i,0)]
                    y1 = y1 + [X.item(i,1)]
                else:
                    x2 = x2 + [X.item(i,0)]
                    y2 = y2 + [X.item(i,1)]
            self._x_range = (max(max(x1,x2)),min(min(x1,x2)))
            self._y_range = (max(max(y1,y2)),min(min(y1,y2)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pts1,pts2,line = ax.plot(x1,y1,x2,y2,[0,2],[0,1])
            plt.setp(pts1, marker="o", c="r", linestyle='')
            plt.setp(pts2, marker="o", c="g", linestyle= '')
            plt.setp(line, c='b', linestyle='-')
            self._line = line


            #TODO prepare graph

        #init the error cache
        self.__update_error_cache()
        
        #main loop for the optimization
        num_changed = 0
        examine_all = True
        alpha = self._alpha
        update = self.__update
        while (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                for i in range(len(alpha)):
                    num_changed += update(i)
            else:
                nonbound = []
                for i in range(len(alpha)):
                    if alpha.item(i,0) != 0 and alpha.item(i,0) != C:
                        nonbound.append(i)
                for i in nonbound:
                    num_changed += update(i)
                    
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return self._b, self._alpha
               
                    
    """
    optimized with respect to index to i2 in the vector alpha and b, it will find
    the second alpha to update.
    @i2, int, index to alpha array
    """
    def __update(self, i2):
        a,b,step,K,y,X,C,u,e,E = self._alpha,self._b,self.__step,self.__K, \
                        self._y,self._X,self._C,self.__u,self._epsilon,self._E

        if self._show_prog:
            print(self._alpha.T.A)
            
        y2 = y.item(i2,0)
        a2 = a.item(i2,0)
        E2 = E[i2]
        r2 = E2*y2
        if ((r2 < -e and a2 < C) or (r2 > e and a2 > 0)):
            nonbound = []
            bound = []
            for i in range(len(a)):
                if (a.item(i,0) != 0 and a.item(i,0) != C):
                    nonbound.append(i)
                else:
                    bound.append(i)
            if (len(nonbound) > 1):
                if E2 > 0:
                    i1 = nonbound[0]
                    min_e = E[i1]
                    for i in range(len(nonbound)):
                        if min_e > nonbound[i]:
                            min_e = nonbound[i]
                            i1 = i
                else:
                    i1 = nonbound[0]
                    max_e = E[i1]
                    for i in range(len(nonbound)):
                        if max_e < nonbound[i]:
                            max_e = nonbound[i]
                            i1 = i
                if (step(i1,i2)):
                    return 1
            for i in nonbound:
                if (step(i,i2)):
                    return 1
            for i in bound:
                if (step(i,i2)):
                    return 1
        return 0

    """
    optimize the function using coordinate descend on 2 variables
    @i1, int, index to alpha array
    @i2, int, index to alpha array
    @return, boolean, true if the step was successful
    @side effects,  the updated vaule for b and elems in i1 and i2
             None means i1 and i2 are bad pair to optimize
    """
    def __step(self, i1, i2):
        m = np.matrix
        u = self.__u
        K = self.__K
        C = self._C
        b = self._b
        e = self._epsilon
        a1, a2 = self._alpha.item(i1,0), self._alpha.item(i2,0)
        x1, x2 = m(self._X.A[i1]).T, m(self._X.A[i2]).T
        y1, y2 = self._y.item(i1,0), self._y.item(i2,0)
        L = max(0, a1+a2-self._C) if y1==y2 else max(0, a2-a1)
        H = min(self._C, a2+a1) if y1==y2 else min(self._C, self._C+a2-a1)

        if L==H or i1==i2:
            return False
        
        eta = K(x1,x1)+K(x2,x2)-2*K(x1,x2)
        if eta <= 0:
            print("warning: eta should be positive")
            return False
        
        E1 = u(x1) - y1
        E2 = u(x2) - y2
        a2_n = a2+y2*(E1-E2)/eta
        if a2_n < L:
            a2_nc = L
        elif a2_n > H:
            a2_nc = H
        else:
            a2_nc = a2_n

        if (abs(a2-a2_nc) < e*(a2+a2_nc+e)):
            return False
        a1_n = a1+y1*y2*(a2-a2_nc)

        b1 = E1 + y1*(a1_n-a1)*K(x1,x1)+y2*(a2_nc-a2)*K(x1,x2)+b
        b2 = E2 + y1*(a1_n-a1)*K(x1,x2)+y2*(a2_nc-a2)*K(x2,x2)+b
        if not (a1_n>0 and a1_n<C):
            b = b1
        elif not (a2_nc>0 and a2_nc<C):
            b = b2
        else:
            b = (b1+b2)/2

        if (self._show_prog and (clock() - self._before) > 1):
            self._before = clock()
            x_big,x1 = self.x_range
            y_big,x2 = self.y_range
            x = []
            step_size1, step_size2 = (x_big-x1)/100.0, (y_big-x2)/100.0
            for i in range(100):
                x.append([x1,x2])
                x1 += step_size1
                x2 += step_size2
            #TODO how to find the line in 2d case????
            
        self._b=b
        self._alpha.A[i1][0]=a1_n
        self._alpha.A[i2][0]=a2_nc
        self.__update_error_cache()
        return True

    def __update_error_cache(self):
        E,u,y,X = self._E,self.__u,self._y,self._X
        for i in range(len(E)):
            E[i] = u(np.matrix(X.A[i]).T)-y.item(i,0)
        

    """
    the kernel, which can be modified on case-by-case basis
    @x1, np.matrix, a vector
    @x2, np.matrix, a vector
    @return, float, the result of the kernel
    """
    def __K(self, x1, x2):
        #customizable
        return (x1.T*x2).item(0,0) #no transformation

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
        
