# Introduction to Machine Learning — teaching notes

Companion to [the interactive HTML lesson](machine_learning.html). The HTML is the concise version to show in class; these notes supply the explanations, examples, and answers that are useful while presenting it. The intended audience knows basic programming but has not studied ML. No derivations or live coding are required.

## Lecture topics and teaching suggestions

| HTML chapter | Main point | Suggested activity |
| --- | --- | --- |
| 01 · Foundations | Learning from examples; learning signals; AI, ML, neural networks, deep learning | Ask what information a delivery-time predictor needs. |
| 02 · Data and preprocessing | Features, targets, preprocessing, dimensionality | Compare nearest neighbors before and after scaling. |
| 03 · Model evaluation | Train / validation / test; cross-validation; regression vs. classification | Select one cross-validation round and explain every block. |
| 04 · Complexity and generalization | Bias, variance, noise, underfitting, overfitting | Compare 1, 4, and 18 tree leaves; change the training sample. |
| 05 · Supervised algorithms | Recognize the main supervised model families | Compare how each method uses the inputs to make a prediction. |
| 06 · Unsupervised learning | K-means; hierarchical clustering; similarity | Step through K-means; compare dendrogram linkages. |

This is an overview, so students should leave able to explain the ideas without implementing every algorithm.

## 1. Foundations of machine learning

A useful opening is: “Instead of listing every rule for estimating delivery time, we show the computer completed deliveries and ask it to fit a relationship.” We still write code: the representation, training procedure, and objective are all designed. What comes from the examples is the fitted model.

**Induction** moves from specific observations toward a general pattern. It does not establish a logically certain rule. Deliveries observed on quiet weekdays may give a poor guide to a busy holiday. **Deduction** applies a general rule to a particular case; its conclusion follows if its premises and reasoning are valid. ML primarily uses inductive learning, although a larger AI system can combine learned models with logical rules and search.

**Generalization** is the reason we care about learning: we want useful predictions on examples outside the training sample. A model can remember its examples and still fail at that job. “Improves with experience” also does not mean every deployed model keeps updating itself. Updating usually requires another deliberate training process.

The nesting diagram distinguishes four terms:

- **AI** is the broader field. Search, planning, and rule-based reasoning can be AI without being ML.
- **ML** fits patterns from data.
- **Neural networks** are one family of ML models.
- **Deep learning** uses neural networks with multiple hidden layers. It is not a separate learning signal.

The learning signals answer a different question:

| Setting | Information supplied during learning | Example |
| --- | --- | --- |
| Supervised | Inputs with targets | Delivery features paired with observed travel time; audio paired with a transcript. |
| Unsupervised | Inputs without target labels | Group deliveries by their characteristics. |
| Semi-supervised | Some labeled examples plus additional unlabeled examples | A small set of labeled images and a large collection of unlabeled images. |
| Reinforcement | Interaction with an environment, observations, and rewards | A game agent choosing moves to improve its expected total reward. |

In reinforcement learning, the learner chooses actions that can change future observations. A reward can be delayed; it is not necessarily a correct answer for each action. A policy describes how the agent chooses actions. Supervised learning usually receives a fixed dataset instead. Semi-supervised learning can help when unlabeled data carries relevant structure, but adding it is not a guarantee of improvement.

### Demonstrations of the learning signals

- **Supervised — Fit and predict:** circles and squares are deliveries with known “on time” and “late” labels. The demo computes the mean feature position of each class and classifies the new diamond by its nearest class mean. The boundary separates those two regions. The diamond is not used for training, and its actual outcome is not supplied.
- **Unsupervised — Find groups:** the points begin without target labels. A small K-means example groups them by feature similarity. Group numbers are assigned by the algorithm; they do not establish that a group means “late” or any other real-world category.
- **Semi-supervised — Use unlabeled structure:** only two points have supplied labels. A graph connects each example to nearby examples, and label information spreads through those connections while the original labels remain fixed. The unlabeled points help describe the geometry. The new labels are inferred, not verified answers. This example works because the graph follows the two curved groups; misleading connections could spread the wrong label.
- **Reinforcement — Train agent:** an agent explores a small grid and updates action values from rewards. Reaching the goal gives +10, a move costs 0.2, and a blocked move costs 0.6. One episode is one attempt, ending at the goal or after a step limit. Each button press runs another set of exploration episodes; the displayed route follows the resulting greedy policy. No target route is provided. The environment is fixed, so reaching this goal demonstrates learning in that environment rather than performance on unfamiliar maps.

For comparison, ask what feedback each learner receives: known answers, feature structure, a few answers plus structure, or rewards from actions. The small induction and deduction flow diagrams distinguish learning a general relationship from examples from applying an already specified rule.

Avoid assigning a percentage of all ML applications to supervised learning. The supplied notebook and HTML give an approximate percentage without establishing what is being counted; it is unnecessary for this lesson.

## 2. Data representation and preprocessing

In the running example, one row is one delivery. Distance, rain, and departure hour are **features**; observed travel time is the **target**. With three feature columns, the input has three dimensions. The dataset has four rows in the introductory table, so it has four examples, not four dimensions.

The target determines the task. Predicting **minutes** is regression. Predicting **late / on time** is classification. A class can be stored as a number without becoming a regression target: “0” and “1” may simply name two categories. Supervised tasks can also have structured outputs, such as transcripts or translations; regression and classification are introductory cases rather than the whole field.

Use $X$ for the feature matrix, $y$ for observed targets, and $\hat y$ for predictions. A model is a fitted mathematical rule, often written $\hat y=f_\theta(x)$, with learned parameters $\theta$. The **algorithm** is the procedure used to fit or apply that rule. “Linear regression” is often used informally for both the model family and its fitting method; distinguish them when needed.

### Preprocessing objectives and common transformations

| Step | Examples used in the HTML | Explanation worth adding |
| --- | --- | --- |
| 01 · Clean | Fill a missing distance with a training median of 4.5 km; investigate −3 km; convert 2,500 m to 2.5 km; unify “Yes,” “Y,” and “yes”; remove confirmed duplicates. | Imputation is an estimate, not a recovered measurement. Negative values need investigation; a long duration can be a legitimate rare event. |
| 02 · Integrate | Join delivery D43's 5 km route record with its 2,000 g parcel record. Flag D44 if its parcel information is missing. Check whether repeated IDs multiply joined rows. | Decide the intended row unit before joining. Multiple scan events per delivery are not automatically duplicate deliveries; aggregate them or use an appropriate join. |
| 03 · Encode and construct | Rain: No → 0, Yes → 1. Vehicle columns [bike, van]: bike → [1, 0], van → [0, 1]. An 18:00 departure gives rush_hour = 1 under an illustrative 17:00–19:00 rule. Extract weekday from the departure date. | Encoding changes the representation; construction derives a new feature. Rush-hour boundaries are a domain choice, not a universal rule. An unseen category needs a defined handling policy. |
| 04 · Scale | A 0–10 km training range maps 5 km to 0.5. A 0–5,000 g range maps 2,500 g to 0.5. With training mean 5 km and standard deviation 2 km, 9 km standardizes to 2. | The standardized value means two standard deviations above the training mean. Estimate every transformation on training data and apply that same transformation to later data. |
| 05 · Aggregate and discretize | Durations 10, 20, 30 minutes become count 3 and mean 20 minutes. Distance bins are short below 3 km, medium from 3 km up to but excluding 7 km, and long from 7 km upward. Thus 2 km is short; 8 km is long. | Aggregation combines observations; discretization replaces precise values with categories. Both discard detail. A historical mean used as a predictor must use only records available before the new prediction. |
| 06 · Reduce | Omit an uninformative delivery ID from predictors, remove a duplicate distance unit, sample 10,000 of 100,000 rows, or replace three correlated parcel dimensions with two PCA components. | Feature selection keeps original columns; PCA creates combinations. Sampling reduces rows, not dimensions. Check rare conditions, retained variance, and predictive performance before accepting the reduction. |

Common missing-value strategies include a training-set median for numeric inputs and a frequent category for categorical inputs. A missingness indicator can also be useful. Dropping rows may be reasonable, but it changes the sample. More elaborate imputers predict missing features from other features. For this overview, explain the choice rather than implementing several imputers.

Do not use the unknown target at prediction time to decide how to impute an input. The PDF mentions class-specific imputation, but applying it naively to future examples would require the answer we are trying to predict. Any supervised preprocessing must respect the same evaluation boundaries as the model.

**Smoothing** attempts to reduce noise, for example by replacing values within a bin with a median or fitting a regression trend. **Binarization** represents a property using two values, often 0 and 1. These are different transformations; none should be applied simply because it is available.

### Demonstration: feature scaling and distance

The new delivery has features **(5 km, 2,000 g)**. The two possible neighbors are:

- A: **(5.2 km, 2,600 g)**.
- B: **(8 km, 2,050 g)**.

Raw Euclidean distances are approximately 600 to A and 50.09 to B. Those calculations mix units, so gram differences dominate. With training ranges 0–10 km and 0–5,000 g, min–max scaling gives:

$$x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}.$$

The scaled distances become approximately **0.122 to A** and **0.300 to B**. The nearest neighbor changes. The displayed coordinate positions stay the same because the plot axes change units too; the numerical distance readout reveals how the weighting changes.

Standardization instead uses:

$$z = \frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}.$$

Standardization does not make a distribution normal, and it is not robust to outliers. Min–max scaling can produce values outside [0, 1] on new data outside the training range. Constant features need special handling to avoid division by zero. Scaling also does not establish that two features deserve equal importance; the representation remains a modeling choice.

**PCA**, mentioned in the PDF, constructs directions that retain as much numeric variance as possible in fewer dimensions. It is unsupervised; large variance need not mean useful information for the target. Scaling can strongly affect the resulting directions. Feature selection keeps some original features; PCA builds combinations of them. No matrix derivation is needed here.

## 3. Training, validation, and test evaluation

Training, validation, and test sets answer different questions:

1. **Training:** what parameter values fit these examples?
2. **Validation:** which model family and settings should we choose?
3. **Test:** how well does the final procedure perform on untouched examples?

The HTML uses 60 / 20 / 20 as an illustration. There is no universal split ratio. Independence and relevance matter more than copying those numbers. Random splitting can be appropriate for independent examples. Use chronological splits for forecasting and grouped splits when multiple rows belong to the same person, route session, or other unit that must remain unseen together. A stratified classification split preserves class proportions when that is appropriate for the task.

**Data leakage** means information enters training or model selection that would not legitimately be available. Examples include using arrival timestamps to predict duration, fitting a scaler to all data before splitting, or putting near-duplicate deliveries into both training and test sets. Good evaluation assesses the entire procedure, including preprocessing.

In **five-fold cross-validation**, the development data is divided into five parts. Each round trains a fresh model on four and validates on one. Every part serves as validation once; average the validation scores to compare settings. Refit preprocessing separately in each round. The separate test set is not a sixth fold. After choosing settings, refit on the available development data and evaluate on the test set. A final test result remains an estimate, with uncertainty from the sample and split.

The diagram uses ordinary folds. Time-dependent data needs a corresponding time-aware validation scheme. Explain the principle rather than adding another diagram during this overview.

### Evaluation metrics and baseline performance

- **Mean absolute error:** $\frac1n\sum_i|y_i-\hat y_i|$. In the delivery example, it is measured in minutes.
- **Mean squared error:** $\frac1n\sum_i(y_i-\hat y_i)^2$. It penalizes larger misses more strongly and is measured in minutes squared.
- **Accuracy:** correct classifications divided by all classifications.

For rare late deliveries, an always-on-time classifier can have high accuracy while being useless for finding delays. If asked, **precision** is the fraction of predicted late deliveries that really are late; **recall** is the fraction of actual late deliveries that were found. Choose measures around the practical cost of mistakes. A simple baseline makes a score easier to interpret: compare with predicting the training-set mean time, or the most common class.

## 4. Model complexity and generalization

**Underfitting** means a model fails to capture important structure, often producing poor training and validation results. **Overfitting** means the fitted solution captures training-specific detail that does not carry over, often producing a gap between strong training performance and weaker validation performance. Distribution shift or evaluation mistakes can also produce a gap, so it is a clue to investigate rather than a complete diagnosis.

**Bias** concerns a systematic difference between the average prediction over possible training samples and the underlying relationship. **Variance** concerns how those predictions change when the training sample changes. Variance is not simply the spread of the observed target values. **Noise** is variation in the target that cannot be explained from the available inputs; collecting an additional relevant feature may make some of it explainable.

For squared-error regression, the classical decomposition expresses expected prediction error as squared bias, variance, and irreducible noise, under the usual assumptions. Students do not need to derive it. The useful lesson is that fitting one sample more closely can come at the cost of sensitivity to which sample was collected.

### Demonstration: regression-tree complexity

1. Set **1 leaf**. A constant prediction misses the changing relationship: a visible underfit.
2. Set about **4 leaves**. The tree follows broad changes. Compare both MSE values.
3. Set **18 leaves**. The tree can isolate the 18 distinct training observations, driving training error to zero. Validation error need not improve.
4. Select **Compare samples**, then **New training sample**. The thin curves come from four additional training samples, not confidence intervals. More variation between predictions illustrates sensitivity to the sample.

The page fits actual one-dimensional regression trees by greedily choosing splits that reduce squared error. It uses noisy data from a known simulated function, so the underlying curve can be shown. Real applications do not supply that curve. Validation data stays fixed; its measured error includes noise. The displayed values are realized sample errors, not exact bias and variance estimates. Do not promise that validation error rises at every added leaf or that four leaves always wins.

Complexity is about representational flexibility, not just a parameter count. Deeper trees, higher-degree polynomials, and smaller K in KNN can all increase flexibility in different ways. The familiar bias–variance trade-off is a useful tendency, not a universal monotonic law. More relevant data, regularization, and suitable model choices can improve both prediction error and stability.

## 5. Supervised learning algorithms

The goal is to explain what each method does with an input. Avoid turning this section into seven separate algorithm lectures.

| Method | Core idea | One limitation or question to mention |
| --- | --- | --- |
| Linear regression | Predict a numeric target with a weighted sum plus an intercept. | Does a linear relationship in the chosen features capture enough of the pattern? |
| Logistic regression | Convert a linear score to a class probability with a sigmoid. | A threshold makes the final decision; a probability estimate is not a guarantee. |
| KNN | Predict from nearby stored examples. | What counts as near, and how many neighbors should vote? |
| SVM | Find a boundary with a useful margin; allow violations when needed. | Scaling, regularization, and kernel choice affect the result. |
| Naive Bayes | Combine class priors and feature likelihoods under conditional independence. | Correlated features can make its probability estimates too confident. |
| Decision tree | Partition inputs using learned questions; predict at leaves. | Unrestricted trees can be unstable and overfit. |
| Ensemble | Combine models by averaging, voting, or sequential correction. | Shared errors do not disappear just because there are many models. |
| Neural network | Learn compositions of weighted transformations and nonlinear activations. | Data, architecture, optimization, and regularization still determine usefulness. |

### Logistic regression: probabilities and decision thresholds

For binary classification,

$$P(y=1\mid x)=\sigma(w^Tx+b),\qquad\sigma(z)=\frac{1}{1+e^{-z}}.$$

In the plot, $z$ is the already computed model score, not necessarily one raw feature. Increasing the score raises the late probability. With $z=1$, the probability is about 73.1%. Raising the threshold from 0.50 to 0.80 changes the predicted class without changing that probability. Choosing a threshold should reflect validation results and mistake costs, not repeated test-set experiments. The usual fitting objective is log loss (binary cross-entropy).

### KNN: instance-based prediction

The displayed KNN uses Euclidean distance, an odd K, and equal neighbor votes. Odd K avoids vote ties for this binary example; it does not solve all possible ties in multiclass KNN. Equal-distance cases in the page use a fixed ordering. A regression version averages neighbor targets, and distance weighting is another possible variant.

KNN is called **non-parametric** because its representational capacity is not limited to a fixed set of fitted coefficients. This does not mean it has no settings: K, the distance metric, and weighting are hyperparameters. Prediction can become expensive with many stored examples. In high dimensions, distances may become less informative, which is one reason feature choice matters.

### SVM: margins, regularization, and kernels

In the separable diagram, support vectors lie on the dashed margin boundaries. In a soft-margin solution, support vectors can also lie inside the margin or be misclassified. Increasing **C** penalizes violations more strongly, which can lead to fitting the training data more tightly. A kernel supplies similarities corresponding to a richer feature representation; it can produce a nonlinear boundary in the original input space. Support vector regression is a related formulation for numeric targets.

### Naive Bayes: conditional independence and class probabilities

Bayes' rule is:

$$P(c\mid x)=\frac{P(x\mid c)P(c)}{P(x)}.$$

The naive factorization replaces $P(x\mid c)$ with a product of feature likelihoods. The assumption is independence **conditioned on the class**, not unconditional independence. It is an approximation rather than a claim about how language actually works.

In the page's toy word example, observing **free** gives unnormalized scores $0.3\times0.7=0.21$ for spam and $0.7\times0.1=0.07$ for not spam. Normalizing gives **75% spam**. Adding **meeting** changes the scores to $0.0105$ and $0.042$, giving **20% spam**. With no selected words, the prior is 30% spam.

This is a small token-likelihood demonstration. It ignores absent words, word counts, the rest of the vocabulary, and smoothing. A Bernoulli text model would also account for absence. Real implementations typically use log probabilities to avoid numerical underflow and smoothing to avoid zero likelihoods. Mention these only if asked. The [scikit-learn Naive Bayes guide](https://scikit-learn.org/stable/modules/naive_bayes.html) explains the conditional-independence assumption and the distinctions between text-model variants.

### Decision trees, ensemble methods, and neural networks

A classification tree chooses splits to reduce impurity, often measured with Gini impurity or entropy. A regression tree reduces a numeric loss, commonly squared error. Prediction follows a path from root to leaf. The model-atlas tree is explicitly hand-built to explain traversal; the earlier overfitting example actually learns split points.

**Bagging** trains models on bootstrap samples and combines them. A **random forest** also considers random feature subsets at splits, helping reduce correlation among trees. Combining varied models can reduce variance. Forest classifiers commonly average class probabilities; the page illustrates this with 20%, 70%, and 80%, whose mean is approximately 57%.

**Boosting** builds a sequence of corrections. AdaBoost gives more weight to previously misclassified observations; gradient boosting fits new learners to loss-related corrections. Do not describe boosting as an automatic cure for overfitting. It also depends on settings such as learning rate and model complexity.

A **perceptron** is a basic linear classifier. A neural network composes layers of weighted sums and activations. Without nonlinear activations between hidden layers, stacking affine transformations still gives an affine transformation. **Backpropagation** computes gradients efficiently; an optimizer uses them to update parameters. Hidden units compute learned activations, rather than being unexplained “unknown variables.”

Deep networks can learn useful features from images, audio, and text. The PDF's driving example illustrates representation learning: road-related image features can emerge from a steering prediction objective. Use that to explain the idea without implying that every hidden unit has a neat human-readable meaning or that deep learning is always the best choice. Neural architectures can also be trained with unsupervised, semi-supervised, or reinforcement signals.

## 6. Unsupervised learning and clustering

Classification has supplied class labels. Clustering has a similarity measure and a grouping procedure. An algorithm can partition data even when there are no distinct natural categories. Ask, “Would changing the features or their units change these groups?” Often it would.

### K-means

K-means searches for K centroids that reduce the sum of squared distances from each point to its assigned centroid:

$$\sum_{i=1}^n \lVert x_i-\mu_{c_i}\rVert^2.$$

The usual alternating procedure is:

1. Initialize K centers.
2. Assign each point to its nearest center.
3. Replace each center with the mean of its assigned points.
4. Repeat assignment and update until a stopping condition is reached.

The HTML deliberately separates assignment and update. Ask what the next button will change before pressing it. Once points are assigned, the displayed squared-distance sum cannot increase during an exact assignment or mean-update step. Convergence does not guarantee the global best solution; starting centers matter. The example uses fixed starting centers so the sequence is repeatable. If a cluster becomes empty, the demo retains its existing center; practical implementations may reinitialize it.

K is a hyperparameter here, but it counts **clusters**, not neighbors. The grouping is affected by scaling and outliers. K-means favors compact groups under squared Euclidean distance and can struggle with elongated groups, unequal densities, or outliers. Repeating initializations and checking whether the result is useful are part of applying it responsibly. There is no target label against which to compute ordinary classification accuracy.

### Hierarchical clustering

**Agglomerative** clustering starts with each point in its own cluster and repeatedly merges the closest pair of clusters. **Divisive** clustering starts with a broad cluster and splits it. A dendrogram records a hierarchy; a horizontal cut can select a grouping at a chosen distance level.

The definition of inter-cluster distance is called **linkage**:

| Linkage | Distance-based definition | Intuition |
| --- | --- | --- |
| Single | Minimum distance between any cross-cluster pair | A close bridge can connect two otherwise spread-out groups; chaining is possible. |
| Complete | Maximum distance between any cross-cluster pair | A merge must account for the farthest cross-pair; tends to favor compact clusters. |
| Average | Mean distance over all cross-cluster pairs | Considers all pairwise links between the two groups. |

The PDF describes these using similarity; the HTML uses **distance**, so “most similar” becomes “nearest.” Higher merge positions in this dendrogram mean **greater dissimilarity**. Average linkage here is the average over all cross-pairs, not the distance between centroids and not an unweighted average over prior cluster summaries.

The five one-dimensional points are A = 0, B = 1, C = 4, D = 6, E = 9.5. Their dendrogram is computed from the selected linkage. With single linkage the final merge distance is 3.5; with complete linkage it is 9.5; with average linkage it is 6.75. Equal-distance ties use a fixed left-to-right ordering; another valid tie choice can produce a different hierarchy. Leaf positions are arranged for readability; the vertical heights carry the merge-distance meaning. Unlike K-means, building the full agglomerative hierarchy does not require choosing a final K first. The [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) provides a reference for the linkage definitions.

## Source materials and scope of the lesson

The lesson synthesizes these supplied local materials:

- [Lab 06 HTML](../../AI_FMI_course/docs/labs/lab06-intro-ml.html): core terminology, bias and variance, training vocabulary, regression, model families, and K-means.
- [Lab 06 notebook](../../AI_FMI_course/notebooks/06_Introduction_to_ML.ipynb): the same main learning objectives and algorithm overview.
- [Intro ML slides, Boris Velichkov](Intro%20ML.pdf): induction versus deduction; semi-supervised and reinforcement learning; the AI / ML / NN / DL relationship; preprocessing, transformation and reduction; train / validation / test and cross-validation; hierarchical clustering and linkage.

The wording and diagrams are new. The lesson preserves the concepts rather than repeating unsupported popularity claims, overly broad statements, or source-specific examples. Preprocessing options receive a short introduction in the HTML and fuller context here.

All interactive data is synthetic. The visualizations use inline SVG and plain JavaScript, with fixed seeds where randomness is involved; no external assets or services are needed. The page can be opened directly from disk. Use **Hide route** in the header to hide the learning route and give the lesson the full available width; **Show route** restores it. The browser remembers this preference when local storage is available.
