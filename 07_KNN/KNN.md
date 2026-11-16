# KNN
## Introduction concepts
### Global vs Local learning
- Global learning: all the dataset is (could be) used in prediction time
  - DFS, BFS, linear regression, naive bayes
- Local learning: only a part of the dataset can be used in prediction time based on the particular input attributes
  - Beam Search, KNN, Hill Climbing, Genetic algorithms
### Instance based vs Model based learning
- Instance based: the prediction is evaluated directly based on a set of the training data, which typically means that the training data should be stored
- Model based learning: the preduction is evaluated based on a set of parameters which were adjusted based on the training data
### Lazy vs Eager learning
- Lazy learning: most of the computation happens at the 
# K-nearest neighbors

Open [the interactive KNN lab](knn_visualizer.html) in a browser. It is a standalone HTML file with no dependencies or server requirements.

## Suggested demonstrations

1. **Overfitting:** start with the default two-moons experiment. Model A uses k = 1 and Model B uses k = 15. Identify the small prediction islands around corrupted training labels. Compare training accuracy with validation accuracy, then increase training label noise. At k = 1, training accuracy includes each point itself and is therefore 100% for these distinct points.
2. **Underfitting:** choose “Too many neighbors.” Compare k = 5 with k = 31 on the rings and observe the loss of local structure. At k equal to the entire balanced training set, uniform votes tie everywhere; this demo's nearest-voter tie rule then decides, producing different behavior at that extreme.
3. **Distance metrics:** choose “Change the metric.” Keep k and voting identical. Move the shared query near a class boundary and compare Euclidean, Manhattan, and Chebyshev neighborhoods, selected point IDs, and predictions. A different metric does not necessarily improve performance.
4. **Feature scaling:** choose “Feature scaling.” The uninformative x coordinate spans 0–100 while the informative y coordinate spans 0–1. Compare raw and standardized distances, then change x back to 0–1. Standardization statistics use training features only.
5. **Voting:** compare uniform and inverse-distance voting at the same k. Inspect individual distances and weights. With inverse-distance voting, exact matches give all weight to the selected zero-distance points.

The two models always share the training and validation data. Model settings leave the sample fixed; **New sample** changes it reproducibly. The error curve uses Model B's metric, voting, and preprocessing, and evaluates every integer k from 1 to min(51, training size). Click the curve to select Model B's k, or use its slider.

## Core ideas

For a query, KNN measures distance to the training examples, selects the k nearest, and combines their labels. This lab demonstrates binary classification; KNN can also perform regression by averaging target values, optionally with distance weights.

| Setting | Effect to explore |
| --- | --- |
| Small k | More sensitivity to individual points and noisy labels; typically higher variance. |
| Large k | Smoother decisions; can introduce bias by mixing distinct local regions. |
| Euclidean distance | √(Σᵢ (xᵢ − qᵢ)²); circular contours in two equally scaled dimensions. |
| Manhattan distance | Σᵢ \|xᵢ − qᵢ\|; diamond contours in two equally scaled dimensions. |
| Chebyshev distance | maxᵢ \|xᵢ − qᵢ\|; square contours in two equally scaled dimensions. |
| Standardization | (xᵢ − μᵢ) / σᵢ per feature, using statistics fitted on training data. |
| Inverse-distance voting | A nearby point has more influence: weight = 1 / distance. |

Small k does not guarantee overfitting, and large k does not guarantee poor predictions. The appropriate settings depend on the distribution, noise, sample size, representation, and task. The guided examples make common effects visible rather than asserting a universal best k or metric.

## Evaluation and implementation conventions

- The training and validation sets are generated independently. Validation always contains 240 points with original labels. The label-noise control flips only training labels, so validation measures robustness to corrupted training labels rather than the error rate against noisy validation labels.
- Training accuracy is resubstitution accuracy: a point may select itself. It is an optimistic measure, especially for k = 1, and is not leave-one-out evaluation.
- Validation points never vote and never contribute to the mean or standard deviation. Both classes initially have equal counts in each split; flipping training labels may alter the training class balance.
- Tied class scores use the nearest voting point's label. Distance ties are ordered by training point ID. Under inverse-distance voting, selected zero-distance matches receive equal weight and all other selected points receive zero weight.
- Vote shares are descriptive scores, not calibrated probabilities. The nearest-neighbor contour includes all equal-distance locations; the ID tie rule may exclude a boundary point from the k selected voters.
- The plot fits each axis to its displayed range. Its screen aspect ratio and differing feature units mean that pixel distances need not equal model distances. The dashed neighborhood contour accounts for the selected metric and scaling.
- Repeated tuning on this validation set can overfit it. Use cross-validation for model selection and an untouched test set for final evaluation in a real project. Fit preprocessing separately inside each training fold.
- The implementation uses exact brute-force distances and a sampled decision background. It stores examples instead of fitting a parametric boundary. Prediction cost grows with training size, and irrelevant features or high dimensionality can make distance less informative.

The page supports mouse queries, keyboard arrow movement on either focused plot, and numeric query coordinates. **Reset lab** restores the default seed and settings. Everything runs locally and offline.
