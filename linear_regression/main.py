import sys
from res.naive_bayes import NaiveBayse

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:python main.py TRAINING_DATA_PATH TEST_DATA_PATH")
        sys.exit(1)

    training_file = sys.argv[1]
    test_file = sys.argv[2]
    #TODO construct the model and then test
