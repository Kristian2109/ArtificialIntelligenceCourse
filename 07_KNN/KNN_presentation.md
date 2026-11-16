# KNN
## Introduction concepts
### Global vs Local learning
- Global learning: all the dataset is (could be) used in prediction time
  - DFS, BFS, linear regression, naive bayes
  - Good when the same pattern holds across the whole dataset
  - Can use information from other regions when local data is limited
- Local learning: only a part of the dataset can be used in prediction time based on the particular input attributes
  - Beam Search, KNN, Hill Climbing, Genetic algorithms
  - Good when different regions of the dataset follow different patterns
  - Useful for similarity based tasks: recommendations, handwriting recognition
### Instance based vs Model based learning
- Instance based: the prediction is evaluated directly based on a set of the training data, which typically means that the training data should be stored
  - KNN
  - Typically, it is also lazy learning
  - Easy to add new examples without retraining a model
  - Useful when similar examples have similar outputs
- Model based learning: the preduction is evaluated based on a set of parameters which were adjusted based on the training data
  - All the algorithms where parameters are trained: Linear Regression, Neural networks...
  - Can summarize the training data in a smaller set of parameters
  - Useful when the training data is too large to store at prediction time
### Lazy vs Eager learning
- Lazy learning: most of the computation happens in prediction time
  - KNN
  - Lazy Model-Based: Locally Weighted regression, it stores the examples and builds a local classifier based on them
  - Little initial training, easy to include new examples
  - Useful when only a few predictions are needed
- Eager learning: most of the computation happens in training time
  - Most of the algorithms which learn parameters because the parameters are learned in advance and the ammount of the parameters is much less than the tranining data
  - Usually faster predictions because most of the work is done in advance
  - Useful when many predictions are needed from the same trained model

## KNN explanation and core ideas
### How it works
Stores the whole training data in memory and in prediction time the result is evaluated only based on the **k** nearest examples
- Classification: majority voting
- Regression: typically weigthed average


### Adjusting the k
- Variance-bias tradeoff: low k increases the training data accuracy, but does not generalize well
### Other parameters
- Weighting the neighbors: typically uniform weighting is used
- Metric choice
    - Euclidean distance
    - Manhattan distance

## KNN benefits and drawbacks
### Benefits
- Not parametrized learning method, so it doesn't take additional assumptions about the data
### Drawbacks
- Curse of dimensionality: increasing the number of attributes requires increasing the training data exponentially, so for each additional attribute the training data should be twice as much

## KNN optimization

## Metrics
### Accuracy
**Properly predicted divided by all**
### Precision
**Properly predicted from a given class divided by the properly predicted plus the incorrectly predicted**???
### Recall
**Not sure also**