#!/usr/bin/python
import sys
from res.naive_bayes import NaiveBayse

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage:main <training_data.mbcj> <testing_data.mbcj>")
        sys.exit(1)
        
    training_file = sys.arvg[1]
    test_file = sys.argv[2]
    naive_bayse = NaiveBayse(training_file)

    (test_data, _) = naive_bayse.parse(test_file)
    correct_num = 0
    incorrect_num = 0
    for test_case in test_data:
        r = naive_bayse.predict(test_case[1])
        if test_case[0] == r:
            correct_num += 1
            print("correctly classified\n")
        else:
            incorrect_num += 1
            print("mis-classified\n")
    print("%d%% out of %d were correctly classified!"
          % (correct_num*100/(correct_num+incorrect_num), correct_num+incorrect_num))
