import numpy as np

"""
parse the data and return result vector and the data matrix, assume 
data format is csv with first column being output
@data_path: string, the absolute data path to the data
@return, (names, y, X), both y, X type np.matrix and names is an array of
         string. y is a vector, X is matrix with row being each data, 
         and column being features, names is the names for each columns.
"""
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



