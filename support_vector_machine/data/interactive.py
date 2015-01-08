import matplotlib.pyplot as plt
import numpy as np
from time import clock

def parse(data_path):
    f = open(data_path)
    line = f.readline()
    names = line[:-1].split(",")
    X, y = [], []
    for line in f:
        if ',' not in line:
            break;
        row = [float(a) for a in line.split(",")]
        y.append([row.pop(0)])
        X.append(row)                       
    return names, np.matrix(y), np.matrix(X)

def f(x_arr, slope):
    y = []
    for x in x_arr:
        y = y + [x*slope]
    return y
        
        

if __name__ == "__main__":
    plt.ion()
    
    _, y, X = parse("./svm_data_2d_with_noise.csv")
    x1,y1,x2,y2 = [],[],[],[]
    for i in range(len(X)):
        if y.item(i,0) == -1:
            x1 = x1 + [X.item(i,0)]
            y1 = y1 + [X.item(i,1)]
        else:
            x2 = x2 + [X.item(i,0)]
            y2 = y2 + [X.item(i,1)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pts1,pts2,line = ax.plot(x1,y1,x2,y2,[0,2],[0,1])
    
    plt.setp(pts1, marker="o", c="r", linestyle='')
    plt.setp(pts2, marker="o", c="g", linestyle= '')
    plt.setp(line, c='b', linestyle='-')

    x = [0, 2]
    tmp = 0
    slopes = []
    for i in range(100):
        tmp = tmp+0.1
        slopes = slopes + [tmp]
    for s in slopes:
        y = f(x,s)
        line.set_ydata(y)
        line.set_xdata(x)
        fig.canvas.draw()
    tmp = input("enter to terminate")

    
    

    
