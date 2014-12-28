import sys
import numpy as np
from LinearReg.linear_reg import LinReg

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:python main.py TRAINING_DATA_PATH TEST_DATA_PATH")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    model = LinReg(training_file)
    names, y, X = model.parse(test_file)
    y_predicted = []
    for x in X.tolist():
        y_result = model.predict(x)
        y_predicted.append(y_result)
        print("with data %s, the output is %f" % (str(x), y_result))

    sum_sq_err = 0.0
    y = y.getA1().tolist()
    for i in range(len(y)):
        sum_sq_err += (y[i]-y_predicted[i])**2
    print("the average error is %f of the test set" % (sum_sq_err/len(y)))
    
    
    
    
