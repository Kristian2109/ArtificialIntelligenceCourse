# KNN
## Introduction concepts
### Global vs Local learning
- Global learning: all the dataset is (could be) used in prediction time
  - DFS, BFS, linear regression, naive bayes
- Local learning: only a part of the dataset can be used in prediction time based on the particular input attributes
  - Beam Search, KNN, Hill Climbing, Genetic algorithms
### Instance based vs Model based learning
- Instance based: the prediction is evaluated directly based on a set of the training data, which typically means that the training data should be stored
  - KNN
  - Typically, it is also an eager learning
- Model based learning: the preduction is evaluated based on a set of parameters which were adjusted based on the training data
  - All the algorithms where parameters are trained: Linear Regression, Neural networks...
### Lazy vs Eager learning
- Lazy learning: most of the computation happens in prediction time
  - KNN
  - Lazy Model-Based: Locally Weighted regression, it stores the examples and builds a local classifier based on them
- Eager learning: most of the computation happens in training time
  - Most of the algorithms
