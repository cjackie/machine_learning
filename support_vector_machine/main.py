import sys
import numpy as np
from res.svm import SVM
from res.util import parse

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("consult README for usage.")
        sys.exit(-1)

    training_data = sys.argv[1]
    test_data = sys.argv[2]
    C = float(sys.argv[3])
    epsilon = float(sys.argv[4])
    show_prog = True if int(sys.argv[5]) == 1 else False

    model = SVM(training_data, C, epsilon, show_prog)
    _, y, X = parse(test_data)

    correct = 0
    incorrect = 0
    for i in range(len(y)):
        if model.predict(np.matrix(X.A[i]).T) == y.item(i,0):
            correct += 1
        else:
            incorrect += 1
    print("%d out of %d were correctly classified" % (correct, correct+incorrect))
