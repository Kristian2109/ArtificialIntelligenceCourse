# KNN and Metrics

## Introduction concepts

### Global vs Local learning

- Global learning: learns one predictive function from the whole training dataset
  - Linear regression, naive Bayes, neural networks
  - Good when the same pattern holds across the whole dataset
  - Can use information from other regions when local data is limited
- Local learning: a prediction depends mainly on examples near the particular input
  - KNN, locally weighted regression
  - Good when different regions of the dataset follow different patterns
  - Useful for similarity based tasks: recommendations, handwriting recognition

### Instance based vs Model based learning

- Instance based: predicts directly from stored training examples
  - KNN
  - Typically also lazy learning
  - Easy to add new examples without retraining a model
  - Useful when similar examples have similar outputs
- Model based: predicts using a model fitted to the training data
  - Linear regression, neural networks, decision trees
  - Can summarize the training data in a smaller set of parameters
  - Useful when the training data is too large to store at prediction time

### Lazy vs Eager learning

- Lazy learning: defers most learning work until prediction time
  - KNN
  - Lazy model based example: locally weighted regression fits a local model when queried
  - Little initial training, though preprocessing and building a search index can take time
- Eager learning: fits a predictive model before receiving queries
  - Linear regression, neural networks, decision trees
  - Usually faster predictions because the model is already fitted
  - Useful when many predictions are needed from the same trained model

## KNN explanation and core ideas

### How it works

- Stores training examples and their targets; supports classification and regression
- At prediction time: measure distances, select the **k nearest neighbors**, combine their targets
  - Finding neighbors may require checking all examples; only the selected neighbors contribute to the prediction
- **Classification:** choose the class with the most votes, or the largest total weight

  $$\hat y(x)=\arg\max_c\sum_{i\in N_k(x)}w_i\,\mathbf{1}[y_i=c]$$

  - Example: five equal votes A, A, A, B, B predict **A**
- **Regression:** average the neighbors' target values, optionally using weights

  $$\hat y(x)=\frac{\sum_{i\in N_k(x)}w_i y_i}{\sum_{i\in N_k(x)}w_i}$$

  - Example: equal weights for targets 10, 12, 20 predict **14**
- $N_k(x)$ is the selected neighborhood; $y_i$ is a target; $w_i$ is its vote weight
  - $\mathbf{1}[y_i=c]$ is 1 when example $i$ belongs to class $c$, otherwise 0

### Adjusting the k

- **Small k:** lower bias and higher variance; flexible boundaries, more sensitive to noise
  - At k = 1, a mislabeled example can change nearby predictions
  - Training accuracy can be perfect when examples match themselves; this does not measure generalization
- **Large k:** higher bias and lower variance; smoother predictions, but can miss local patterns
  - If k includes all training examples, uniform voting predicts the most common class; regression predicts the overall mean
- Select k using validation or cross-validation; reserve a separate test set for final evaluation
  - Odd k avoids uniform vote ties for two classes, but not necessarily for multiple classes

### Other parameters

- **Neighbor weights:** uniform weighting is the usual starting point
  - Uniform: $w_i=1$; every selected neighbor contributes equally
  - Distance weighted: for example, $w_i=1/(d(x,x_i)+\varepsilon)$, with a small $\varepsilon>0$ to avoid division by zero
  - Useful when closer neighbors should matter more; can increase sensitivity to nearby noise
- **Distance metric:** defines similarity; $p$ is the number of features
  - Euclidean: $d(x,z)=\sqrt{\sum_{j=1}^{p}(x_j-z_j)^2}$; common for scaled numeric features
  - Manhattan: $d(x,z)=\sum_{j=1}^{p}|x_j-z_j|$; total absolute difference, with less emphasis on a single large coordinate difference
  - Chebyshev: $d(x,z)=\max_j|x_j-z_j|$; useful when the largest coordinate difference determines similarity
### Feature scaling
Large numeric ranges can dominate distances
- Standardization: $x'_j=(x_j-\mu_j)/\sigma_j$; min–max scaling is another option
- Fit scaling on training data only, including within each cross-validation fold; reuse it for validation and test data
- Encode categorical features appropriately; arbitrary category numbers create misleading distances

## KNN benefits and drawbacks

### Benefits

- Simple to understand; predictions can be explained through their neighbors
- **Non-parametric:** no fixed form such as a straight line is imposed on the relationship
  - Still assumes the chosen distance makes neighbors relevant; k, weights, and metric are hyperparameters
- Captures nonlinear patterns with little model fitting; works for classification and regression

### Drawbacks

- Stores the training set; memory use and prediction time grow with the number of examples
- Sensitive to scaling, irrelevant features, noisy labels, and outliers
- Unequal class frequencies or densities can let common classes dominate local votes
- **Curse of dimensionality:** as features increase, data becomes sparse and distances can become less informative
  - Comparable coverage can require exponentially more data; there is no universal “double the data per feature” rule
  - Feature selection or dimensionality reduction can help

## KNN optimization

- **Brute force:** compare a query with every training example
  - Distance calculations cost $O(np)$ per query for n examples and p features; often effective for small datasets
- **KD-tree (k-dimensional tree):** organizes training points in a binary tree to reduce distance comparisons
  - **Build:** split points by one feature, usually at its median, then repeat inside each group. In 2D, alternate vertical cuts on x and horizontal cuts on y
  - **Search:** follow the splits to the query's region, calculate point distances, and keep the k closest candidates found so far
  - **Backtrack:** check other branches too, because a nearest neighbor may lie across a split
  - **Prune:** once k candidates are available, let r be the distance to the farthest candidate. Skip a branch if its entire region is farther than r from the query
  - **Example (Euclidean distance):** a split is at x = 5, the query has x = 2, and r = 1. Every point on the x ≥ 5 side is at least 3 units away, so that branch can be skipped
- Building a KD-tree costs time and memory; it is often useful in low dimensions but can approach brute-force work in high dimensions
  - Exact search preserves the KNN prediction rule; the tree changes how neighbors are found
- For very large datasets, approximate nearest-neighbor search trades some neighbor accuracy for speed

# Metrics

- Evaluate predictions on held-out data; use classification metrics for class labels and regression metrics for numeric targets

# Categorical prediction metrics

- For a chosen positive class: **TP** = correctly predicted positive, **TN** = correctly predicted negative, **FP** = false alarm, **FN** = missed positive

## Confusion matrix

- One row and column per **class**: rows are actual labels; columns are predicted labels
- The **diagonal** contains correct predictions; other cells show which classes were confused

**Two classes: spam and not spam** — spam is the positive class

| Actual ↓ / Predicted → | Not spam | Spam |
| --- | ---: | ---: |
| Not spam | **18 (TN)** | 2 (FP) |
| Spam | 4 (FN) | **6 (TP)** |

- Of 10 actual spam emails, 6 were found and 4 were missed
- Of 8 emails flagged as spam, 6 really were spam and 2 were false alarms
- Accuracy = diagonal / total = (18 + 6) / 30 = **80%**

**Three classes: A, B, and C**

| Actual ↓ / Predicted → | A | B | C |
| --- | ---: | ---: | ---: |
| A | **8** | 1 | 1 |
| B | 2 | **6** | 2 |
| C | 1 | 0 | **9** |

- Example: the 2 in row B, column A means **2 actual B examples were predicted as A**
- Accuracy = (8 + 6 + 9) / 30 ≈ **76.7%**
- Calculate precision and recall for each class by treating it as positive and all other classes as negative
  - For A: TP = 8, FP = 2 + 1 = 3, FN = 1 + 1 = 2, TN = 17
  - Precision = 8/11 ≈ **72.7%**; recall = 8/10 = **80%**

## Accuracy

- **Correct predictions divided by all predictions**
  - $\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$
  - Can hide poor rare-class performance: always predicting “not spam” gives 99% accuracy if only 1% of emails are spam

## Precision

- **Of the examples predicted positive, how many really are positive?**
  - $\text{Precision}=\frac{TP}{TP+FP}$
  - Emphasize precision when false alarms are costly, such as legitimate emails marked as spam

## Recall

- **Of the actual positive examples, how many did we find?**
  - $\text{Recall}=\frac{TP}{TP+FN}$
  - Emphasize recall when missed positives are costly, such as fraud left undetected

## F1 score

- **Harmonic mean of precision and recall**; high only when both are high
  - $F_1=\frac{2\,\text{Precision}\,\text{Recall}}{\text{Precision}+\text{Recall}}=\frac{2TP}{2TP+FP+FN}$
  - Ranges from 0 to 1; higher is better. Useful when precision and recall both matter, especially for a rare positive class
- In the spam example: precision = 6/8 = **75%**, recall = 6/10 = **60%**, F1 = 12/18 ≈ **66.7%**
- For multiple classes, calculate each metric per class; **macro averaging** gives every class equal weight

# Regression metrics

- Measure how far numeric predictions are from actual values
  - $n$ = number of evaluated examples, $y_i$ = actual value, $\hat y_i$ = prediction, $\bar y$ = mean actual value on the evaluation set

## MAE — Mean Absolute Error

- **Average absolute prediction error**
  - $\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|$
  - In the target's original units; an MAE of 3 minutes means predictions miss by 3 minutes on average
  - Useful for a directly interpretable error measure; less sensitive to large errors than squared-error metrics

## MSE — Mean Squared Error

- **Average squared prediction error**
  - $\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2$
  - Squaring gives large errors more influence; useful when large misses should be penalized more heavily
  - Measured in squared target units, such as minutes squared

## RMSE — Root Mean Squared Error

- **Square root of MSE**
  - $\text{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}$
  - Emphasizes large errors while returning to the target's original units
  - Example: absolute errors of 1, 1, and 4 give MAE = **2**, MSE = **6**, and RMSE ≈ **2.45**
- For MAE, MSE, and RMSE, **lower is better**; 0 means perfect predictions

## R² — Coefficient of determination

- **Compares squared prediction error with always predicting the evaluation-set mean**
  - $R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat y_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}$
  - For nonconstant targets: **1** = perfect predictions, **0** = matches the mean baseline, **negative** = worse than that baseline
  - Unitless; higher is better. Read alongside MAE or RMSE to understand the error in practical units

Reference: scikit-learn guides to [nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics), and [regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
