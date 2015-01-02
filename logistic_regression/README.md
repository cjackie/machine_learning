### usage
```shell
cd logistic_regression
python main.py TRAINING_DATA_PATH TEST_DATA_PATH OPTIMIZATION_CODE ALPHA RESOLUTION SHOW_PROGRESS
```

### optimization code
| optimization code | 1                          | 2               | 3             |
|-------------------|----------------------------|-----------------|---------------|
| method            | stochastic gradient ascend | gradient ascend | newton method |

### alpha
alpha specifies the "step size" for stochastic gradient ascend and gradient ascend. It specifies how much the optimization algorithm to increase along the gradient. Don't set it too high, otherwise it will step over the maximum. 0.1 or also is worth to try.

### resolution
resolution specifies how accuracy the maximum should. In the algorithm this quantity will be compare to the gradient each time to determine if the max is reached. typical resolution number can be 10^-5

### show_progress
| show_progress | 0     | 1    |
|---------------|-------|------|
| meaning       | false | true |
no supprise...<br>
if set it to true, you will able to see the theta and data for each optimization iteration. Good for seeing what's going on and get insights.
