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

Open [the KNN examples](knn_visualizer.html) in a browser. The standalone HTML runs offline, with no dependencies or server requirement.

## Seven graphs on one page

1. **k = 1:** Euclidean decision regions show sensitivity to individual points and noisy labels.
2. **k = 15:** A larger vote often smooths small prediction islands.
3. **k = 51:** Broad neighborhoods can lose local class structure.
4. **Euclidean neighbors:** Nine voters selected using √(Δx² + Δy²).
5. **Manhattan neighbors:** Nine voters selected using |Δx| + |Δy|.
6. **Chebyshev neighbors:** Nine voters selected using max(|Δx|, |Δy|).
7. **Error versus k:** Training and validation error for Euclidean KNN, for every integer k from 1 to 51.

There are only two shared controls: **Training label noise** and **New sample**. The first three graphs keep the distance metric fixed and change k. The next three keep k = 9 fixed and change the metric. Click any metric graph to move the query in all three, or focus a graph and use the arrow keys. Shift plus arrow moves farther. No connections or extra overlays are drawn; the metric graphs emphasize just the nine voters and their neighborhood outline.

## Suggested demonstrations

- Compare the first row at 15% label noise. Small k memorizes individual labels; a larger k may improve validation accuracy even while training accuracy falls. Set noise to 0% and compare again.
- Follow the validation error curve. The lowest error can occur between the three illustrated k values. Generate another sample to see that a useful k depends on the data.
- Click near a boundary in the metric row. Compare which points are highlighted and how their votes change. A different set of neighbors need not produce a different class or improve accuracy.

## Core ideas and conventions

KNN predicts from stored training examples rather than fitting a parametric boundary. Small neighborhoods are often sensitive to noise; very large neighborhoods may mix distinct regions. These are common effects, rather than guarantees about a particular k.

All seven graphs share 100 synthetic two-moons training points. The 240 validation points are generated independently, never vote, and retain their original labels. Label noise flips exactly that percentage of training labels; changing noise keeps the feature positions and validation data fixed. **New sample** changes both sets reproducibly.

Training accuracy includes self-matches: each training point can find itself, so k = 1 achieves 100% training accuracy on these distinct examples, including corrupted labels. This is resubstitution accuracy, not leave-one-out evaluation. Validation accuracy measures performance against original labels after learning from possibly corrupted training labels.

Voting is uniform. Class-vote ties use the nearest training point's label; distance ties use point ID. The six illustrated classifiers use odd k. The error curve also evaluates even k with the stated tie rule. Equal-distance points can lie on a neighborhood contour without all being among the nine selected voters.

The distance calculations use raw feature coordinates. The plotted axes fill the available area, so contours may appear stretched on screen. In a real problem, feature units and scaling affect distances; any preprocessing should be fitted only on training data.

Repeated tuning on validation can overfit that set. Use cross-validation for model selection and a separate untouched test set for final evaluation. The implementation uses exact brute-force neighbors and a sampled decision background; all figures and scores are computed locally.
