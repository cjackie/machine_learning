import sys
import logistic_reg.log_reg.LogReg

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage:python main.py TRAINING_DATA_PATH \
               TEST_DATA_PATH OPTIMIZATION_CODE ALPHA SHOW_PROGRESS")
        print("consult README file details about optimization code, alpha \
               and show_progress")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    optimiztion = int(sys.argv[3])
    alpha = float(sys.argv[4])
    #TODO
