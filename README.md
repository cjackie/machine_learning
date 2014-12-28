machine_learning
================

implementations of machine learning algoithms<br>
TODO list:
- [x] Naive Bayes classifier
- [x] Linear Regression
- [ ] Logistic Regression
- [ ] Support Vector Machines
- [ ] K-means Clustering

### Naive Bayes classifier
Naive Bayes assumes that input features are independent in probability, which is a strong assumption. The size of input features is fixed, which can be very large(more than 1000s). But the output is just binary. Naive Bayes classifier uses the bayse rule as the model to make predictions.

### Linear Regression
The most simple yet powerful model for mapping data. It assumes the outputs are linearly depended on data with gaussian noise. The data itself can be transformed or even mapped to higher dimensions, and the same algorithm can be applied.

### Logistic Regression
A model with the output being only 2 possible states(binary output).

### Support Vector Machines
A newest machine learning algorithm which appeared around 1990s. This algorithm is also a classification algorithm that ouputs binary value. It seperates two categories with largest margin. The result is very elegant, because the data can be transformed to other dimensions(even infinite dimesion) in linear time O(n) by using [kernel method](http://en.wikipedia.org/wiki/Kernel_method), which means high effiency in computing the result. I've read the algorithm several times, and I still can't fully understand it, but I do know how to implement it...The notion of infinite dimension(using gaussian kernel) is very interesting!!!

