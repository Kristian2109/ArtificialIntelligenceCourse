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

Open [the interactive examples](knn_visualizer.html) in a browser. The page runs offline. Compact, wide graphs use the available horizontal space; the linked k map and accuracy curve sit next to each other on larger screens and stack on smaller screens.

## Examples

1. **k and accuracy:** one slider updates decision regions, plotted validation mistakes, training accuracy, and validation accuracy/error. The accuracy curve and map use the same sample. Click the curve to select k as well.
2. **Distance metrics:** switch Euclidean, Manhattan, or Chebyshev with k = 9 fixed. The neighborhood, query prediction, validation predictions, and correct count update. Click the map to move the query; arrow keys also work when it is focused.
3. **Label noise:** keep k = 1 and corrupt training labels. Feature positions and validation labels stay fixed. Compare the new prediction islands and red validation errors. Training accuracy includes self-matches.
4. **Feature scaling:** compare raw and standardized distances with k = 9. The irrelevant x feature spans 0–100; informative y spans 0–1. Training and validation data stay fixed, and preprocessing uses training statistics only.
5. **KNN regression:** below classification, change k from 1 to 30. Predictions average the k nearest training targets using x distance only. The piecewise constant prediction curve, train RMSE, and validation RMSE update. All 30 training and 60 validation observations are plotted. Lower RMSE is better.

## Data and evaluation

Classification uses 80 training and 120 validation examples. A wavy boundary, y = 0.5 + 0.22 sin(2πx), defines the first three examples. Independently jittered training and validation grids spread points across the plot and make accuracy easier to inspect. The k example flips 15% of training labels (12 of 80), the metric example is clean, and the label-noise example varies the corruption. Feature scaling uses uncorrupted horizontal bands. All samples are seeded and reproducible.

Filled circles are training points. Hollow diamonds are every validation point, colored by its true label. Red crosses mark incorrect validation predictions. The displayed count and accuracy use exactly these points. A dashed reference boundary shows the synthetic labeling rule. Validation points never vote and never contribute to preprocessing.

Training accuracy includes self-matches and is deliberately optimistic. Voting is uniform; class-vote ties use the nearest training point and equal distances use point ID. The k chart includes every integer from 1 to 51 with this tie rule. Decision backgrounds are sampled while scores and neighbors are exact. Wide plots fit each axis independently, so distance contours may look stretched.

Regression uses y = sin(0.8x) + 0.04x + ε, with 30 training and 60 independent validation targets and Gaussian noise of standard deviation 0.2. Prediction averages training targets only. RMSE is √mean((prediction − observed target)²), evaluated against the plotted noisy observations. The displayed step curve uses pairwise training-x midpoints to locate every possible neighbor change. At k = 30 it becomes the global training-target mean; at k = 1 training RMSE is zero for these distinct x values.

Small k can overfit and large k can underfit, but neither is inevitable. Repeated tuning can overfit validation. Use cross-validation for model selection and an untouched test set for final evaluation.
