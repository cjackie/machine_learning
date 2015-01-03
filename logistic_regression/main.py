import sys
import numpy as np
from logistic_reg.log_reg import LogReg
from logistic_reg.util import parse

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("usage:python main.py TRAINING_DATA_PATH " + \
              "TEST_DATA_PATH OPTIMIZATION_CODE ALPHA RESOLUTION SHOW_PROGRESS")
        print("consult README file details about optimization code, alpha" + \
              "and show_progress")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    optimization = int(sys.argv[3])
    alpha = float(sys.argv[4])
    resolution = float(sys.argv[5])
    show_prog = True if sys.argv[6]=='1' else False
    opt_code = 0
    if optimization == 1:
        opt_code=LogReg.STOCHASTIC_GRADIENT_ASCENT
    elif optimization == 2:
        opt_code=LogReg.GRADIENT_ASCENT
    elif optimization == 3:
        opt_code=LogReg.NEWTON_METHOD
    model = LogReg(training_file, resolution, alpha, opt_code, show_prog)

    (names, y, X) = parse(test_file)
    (correct_counts, incorrect_counts) = (0,0)
    for i in range(len(X.A)):
        ans = model.predict(np.matrix(X.A[i]).T)
        if ans>0.5:
            correct_counts += 1 if y.A[i][0]==1 else 0
            incorrect_counts += 1 if y.A[i][0]==0 else 0
        else:
            correct_counts += 1 if y.A[i][0]==0 else 0
            incorrect_counts += 1 if y.A[i][0]==1 else 0

    print("%f%% of the data were corrected classified!" % \
          (float(correct_counts)/(correct_counts+incorrect_counts)*100))
    
        
        
        
