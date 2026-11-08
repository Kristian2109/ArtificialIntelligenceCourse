# Introduction to Machine Learning — teaching notes

Companion to [the interactive HTML lesson](machine_learning.html). The HTML is the concise version to show in class; these notes supply the explanations, examples, and answers that are useful while presenting it. The intended audience knows basic programming but has not studied ML. No derivations or live coding are required.

## A 90-minute teaching plan

| Time | HTML chapter | Main point | Suggested activity |
| --- | --- | --- | --- |
| 00–08 | 01 · The idea | Learning from examples; learning signals; AI, ML, neural networks, deep learning | Ask what information a delivery-time predictor needs. |
| 08–20 | 02 · Make data useful | Features, targets, preprocessing, dimensionality | Compare nearest neighbors before and after scaling. |
| 20–32 | 03 · Keep an honest score | Train / validation / test; cross-validation; regression vs. classification | Select one cross-validation round and explain every block. |
| 32–44 | 04 · How training works | Model, parameters, objective, hyperparameters, optimization, batches | Adjust a regression line, then fit it. Change batch size. |
| 44–56 | 05 · Fit vs. generalize | Bias, variance, noise, underfitting, overfitting | Compare 1, 4, and 18 tree leaves; change the training sample. |
| 56–74 | 06 · Meet the models | Recognize the main supervised model families | Spend about two minutes per method, with time for transitions. |
| 74–85 | 07 · Find hidden groups | K-means; hierarchical clustering; similarity | Step through K-means; compare dendrogram linkages. |
| 85–90 | 08 · Check your intuition | Explain a task, a model choice, and an evaluation result | Answer the three checks and discuss the reasons. |

The activities are included in these times. Treat the extra model names in the expandable HTML section as a brief signpost; they do not need separate demonstrations. If discussion runs long, shorten the model walkthrough rather than removing the evaluation section. This is an overview, so students should leave able to explain the ideas without implementing every algorithm.

## 1. What does “learning” mean here?

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

Avoid assigning a percentage of all ML applications to supervised learning. The supplied notebook and HTML give an approximate percentage without establishing what is being counted; it is unnecessary for this lesson.

## 2. From raw observations to model inputs

In the running example, one row is one delivery. Distance, rain, and departure hour are **features**; observed travel time is the **target**. With three feature columns, the input has three dimensions. The dataset has four rows in the introductory table, so it has four examples, not four dimensions.

The target determines the task. Predicting **minutes** is regression. Predicting **late / on time** is classification. A class can be stored as a number without becoming a regression target: “0” and “1” may simply name two categories. Supervised tasks can also have structured outputs, such as transcripts or translations; regression and classification are introductory cases rather than the whole field.

Use $X$ for the feature matrix, $y$ for observed targets, and $\hat y$ for predictions. A model is a fitted mathematical rule, often written $\hat y=f_\theta(x)$, with learned parameters $\theta$. The **algorithm** is the procedure used to fit or apply that rule. “Linear regression” is often used informally for both the model family and its fitting method; distinguish them when needed.

### What preprocessing is trying to accomplish

| Step | Concrete example | Explanation worth adding |
| --- | --- | --- |
| Cleaning | Missing rain flag, a negative distance, mixed “yes” / “Y” codes | Fix known errors; investigate missingness and unusual values. An extreme value can be a valid rare event. |
| Integration | Join route data and parcel data by delivery ID | Incorrect joins and duplicate records can change the effective sample or leak information between splits. |
| Encoding | Represent a rain flag as 0/1; one-hot encode unordered vehicle types | Giving arbitrary integers to unordered categories can create a misleading order or distance. |
| Feature engineering | Convert departure time into a rush-hour indicator | Use domain knowledge to make relevant information easier to use. The feature must be available when predicting. |
| Scaling | Put distance and parcel mass on comparable scales | Especially relevant to distance-based methods and many optimization procedures. Tree splits generally do not require it. |
| Aggregation / generalization | Summarize delivery events by day; replace a detailed vehicle subtype with a broader category | Changes the resolution of the data. It may make patterns easier to estimate while discarding detail. |
| Discretization | Turn minutes into short / medium / long intervals | Replaces a numeric feature with categories. Boundaries matter; binning is not the same as scaling. |
| Reduction | Select features, project numeric features with PCA, or sample rows | Reduces input dimensions or data volume. Check what information is lost. |

Common missing-value strategies include a training-set median for numeric inputs and a frequent category for categorical inputs. A missingness indicator can also be useful. Dropping rows may be reasonable, but it changes the sample. More elaborate imputers predict missing features from other features. For this overview, explain the choice rather than implementing several imputers.

Do not use the unknown target at prediction time to decide how to impute an input. The PDF mentions class-specific imputation, but applying it naively to future examples would require the answer we are trying to predict. Any supervised preprocessing must respect the same evaluation boundaries as the model.

**Smoothing** attempts to reduce noise, for example by replacing values within a bin with a median or fitting a regression trend. **Binarization** represents a property using two values, often 0 and 1. These are different transformations; none should be applied simply because it is available.

### The scaling demonstration

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

## 3. Evaluation: decide what “new” should mean

Training, validation, and test sets answer different questions:

1. **Training:** what parameter values fit these examples?
2. **Validation:** which model family and settings should we choose?
3. **Test:** how well does the final procedure perform on untouched examples?

The HTML uses 60 / 20 / 20 as an illustration. There is no universal split ratio. Independence and relevance matter more than copying those numbers. Random splitting can be appropriate for independent examples. Use chronological splits for forecasting and grouped splits when multiple rows belong to the same person, route session, or other unit that must remain unseen together. A stratified classification split preserves class proportions when that is appropriate for the task.

**Data leakage** means information enters training or model selection that would not legitimately be available. Examples include using arrival timestamps to predict duration, fitting a scaler to all data before splitting, or putting near-duplicate deliveries into both training and test sets. Good evaluation assesses the entire procedure, including preprocessing.

In **five-fold cross-validation**, the development data is divided into five parts. Each round trains a fresh model on four and validates on one. Every part serves as validation once; average the validation scores to compare settings. Refit preprocessing separately in each round. The separate test set is not a sixth fold. After choosing settings, refit on the available development data and evaluate on the test set. A final test result remains an estimate, with uncertainty from the sample and split.

The diagram uses ordinary folds. Time-dependent data needs a corresponding time-aware validation scheme. Explain the principle rather than adding another diagram during this overview.

### A score needs context

- **Mean absolute error:** $\frac1n\sum_i|y_i-\hat y_i|$. In the delivery example, it is measured in minutes.
- **Mean squared error:** $\frac1n\sum_i(y_i-\hat y_i)^2$. It penalizes larger misses more strongly and is measured in minutes squared.
- **Accuracy:** correct classifications divided by all classifications.

For rare late deliveries, an always-on-time classifier can have high accuracy while being useless for finding delays. If asked, **precision** is the fraction of predicted late deliveries that really are late; **recall** is the fraction of actual late deliveries that were found. Choose measures around the practical cost of mistakes. A simple baseline makes a score easier to interpret: compare with predicting the training-set mean time, or the most common class.

## 4. Training, optimization, and the words around them

For the one-feature regression model,

$$\hat y=mx+b,$$

$m$ is minutes per kilometer and $b$ is a predicted time at zero distance. Do not assume the intercept has a sensible physical interpretation outside the observed range. The fitted line is an approximation, not a claim that distance is the only cause of delay.

The **residual** is $y-\hat y$: an observed vertical gap in the plot. Ordinary least squares minimizes the sum of squared vertical residuals, not perpendicular distances to the line. Dividing the sum by the number of examples gives MSE and leaves the minimizing parameters unchanged. The page uses six invented deliveries; they are a separate illustrative sample from the four-row feature table.

**Suggested demonstration:** move the line away from the observations and ask why the error grows. Adjust slope and intercept by eye, then press **Least-squares fit**. An optimized line still has nonzero error because the observations do not lie perfectly on one line.

A **parameter** is learned from the training data: slope, intercept, or neural-network weights. A **hyperparameter** is a setting selected around that fitting procedure: learning rate, batch size, maximum tree depth, or K in KNN. Tuning hyperparameters with validation is still part of development, so validation scores are not an untouched final performance estimate.

A **loss** usually measures one example's prediction error. A **cost / objective** aggregates losses, and it may include a regularization penalty. The words are sometimes used interchangeably, but this distinction is helpful.

**Gradient descent** updates parameters in the direction that locally reduces a differentiable objective:

$$\theta\leftarrow\theta-\eta\nabla_\theta L.$$

The learning rate $\eta$ sets the step size. Too small can be slow; too large can oscillate or diverge. This is one optimization method, not the definition of all ML training. Ordinary least squares can be solved directly. A decision tree searches for useful splits. KNN mostly stores examples and does its comparisons at prediction time.

### Batches, iterations, and epochs

With **240 examples** and a **batch size of 24**, one epoch contains **10 parameter updates** in the mini-batch loop shown. Three epochs contain 30 updates. A batch is a group of examples; an iteration is one update in this setting; an epoch is one pass through the full training set.

Keeping a smaller final batch gives $\lceil n/B\rceil$ updates per epoch. For a batch size of 32, there are 8 updates, and the last batch contains 16 examples. Some training systems drop the last incomplete batch; the HTML explicitly keeps it. These definitions describe a common training loop and should not imply that every model learns through epochs.

## 5. Bias, variance, and the temptation to memorize

**Underfitting** means a model fails to capture important structure, often producing poor training and validation results. **Overfitting** means the fitted solution captures training-specific detail that does not carry over, often producing a gap between strong training performance and weaker validation performance. Distribution shift or evaluation mistakes can also produce a gap, so it is a clue to investigate rather than a complete diagnosis.

**Bias** concerns a systematic difference between the average prediction over possible training samples and the underlying relationship. **Variance** concerns how those predictions change when the training sample changes. Variance is not simply the spread of the observed target values. **Noise** is variation in the target that cannot be explained from the available inputs; collecting an additional relevant feature may make some of it explainable.

For squared-error regression, the classical decomposition expresses expected prediction error as squared bias, variance, and irreducible noise, under the usual assumptions. Students do not need to derive it. The useful lesson is that fitting one sample more closely can come at the cost of sensitivity to which sample was collected.

### How to present the tree demonstration

1. Set **1 leaf**. A constant prediction misses the changing relationship: a visible underfit.
2. Set about **4 leaves**. The tree follows broad changes. Compare both MSE values.
3. Set **18 leaves**. The tree can isolate the 18 distinct training observations, driving training error to zero. Validation error need not improve.
4. Select **Compare samples**, then **New training sample**. The thin curves come from four additional training samples, not confidence intervals. More variation between predictions illustrates sensitivity to the sample.

The page fits actual one-dimensional regression trees by greedily choosing splits that reduce squared error. It uses noisy data from a known simulated function, so the underlying curve can be shown. Real applications do not supply that curve. Validation data stays fixed; its measured error includes noise. The displayed values are realized sample errors, not exact bias and variance estimates. Do not promise that validation error rises at every added leaf or that four leaves always wins.

Complexity is about representational flexibility, not just a parameter count. Deeper trees, higher-degree polynomials, and smaller K in KNN can all increase flexibility in different ways. The familiar bias–variance trade-off is a useful tendency, not a universal monotonic law. More relevant data, regularization, and suitable model choices can improve both prediction error and stability.

## 6. A short guide to the model families

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

### Logistic regression: probability and decision are separate

For binary classification,

$$P(y=1\mid x)=\sigma(w^Tx+b),\qquad\sigma(z)=\frac{1}{1+e^{-z}}.$$

In the plot, $z$ is the already computed model score, not necessarily one raw feature. Increasing the score raises the late probability. With $z=1$, the probability is about 73.1%. Raising the threshold from 0.50 to 0.80 changes the predicted class without changing that probability. Choosing a threshold should reflect validation results and mistake costs, not repeated test-set experiments. The usual fitting objective is log loss (binary cross-entropy).

### KNN: the training examples remain part of the predictor

The displayed KNN uses Euclidean distance, an odd K, and equal neighbor votes. Odd K avoids vote ties for this binary example; it does not solve all possible ties in multiclass KNN. Equal-distance cases in the page use a fixed ordering. A regression version averages neighbor targets, and distance weighting is another possible variant.

KNN is called **non-parametric** because its representational capacity is not limited to a fixed set of fitted coefficients. This does not mean it has no settings: K, the distance metric, and weighting are hyperparameters. Prediction can become expensive with many stored examples. In high dimensions, distances may become less informative, which is one reason feature choice matters.

### SVM: explain the margin before kernels

In the separable diagram, support vectors lie on the dashed margin boundaries. In a soft-margin solution, support vectors can also lie inside the margin or be misclassified. Increasing **C** penalizes violations more strongly, which can lead to fitting the training data more tightly. A kernel supplies similarities corresponding to a richer feature representation; it can produce a nonlinear boundary in the original input space. Support vector regression is a related formulation for numeric targets.

### Naive Bayes: say what is assumed independent

Bayes' rule is:

$$P(c\mid x)=\frac{P(x\mid c)P(c)}{P(x)}.$$

The naive factorization replaces $P(x\mid c)$ with a product of feature likelihoods. The assumption is independence **conditioned on the class**, not unconditional independence. It is an approximation rather than a claim about how language actually works.

In the page's toy word example, observing **free** gives unnormalized scores $0.3\times0.7=0.21$ for spam and $0.7\times0.1=0.07$ for not spam. Normalizing gives **75% spam**. Adding **meeting** changes the scores to $0.0105$ and $0.042$, giving **20% spam**. With no selected words, the prior is 30% spam.

This is a small token-likelihood demonstration. It ignores absent words, word counts, the rest of the vocabulary, and smoothing. A Bernoulli text model would also account for absence. Real implementations typically use log probabilities to avoid numerical underflow and smoothing to avoid zero likelihoods. Mention these only if asked. The [scikit-learn Naive Bayes guide](https://scikit-learn.org/stable/modules/naive_bayes.html) explains the conditional-independence assumption and the distinctions between text-model variants.

### Trees, ensembles, and networks

A classification tree chooses splits to reduce impurity, often measured with Gini impurity or entropy. A regression tree reduces a numeric loss, commonly squared error. Prediction follows a path from root to leaf. The model-atlas tree is explicitly hand-built to explain traversal; the earlier overfitting example actually learns split points.

**Bagging** trains models on bootstrap samples and combines them. A **random forest** also considers random feature subsets at splits, helping reduce correlation among trees. Combining varied models can reduce variance. Forest classifiers commonly average class probabilities; the page illustrates this with 20%, 70%, and 80%, whose mean is approximately 57%.

**Boosting** builds a sequence of corrections. AdaBoost gives more weight to previously misclassified observations; gradient boosting fits new learners to loss-related corrections. Do not describe boosting as an automatic cure for overfitting. It also depends on settings such as learning rate and model complexity.

A **perceptron** is a basic linear classifier. A neural network composes layers of weighted sums and activations. Without nonlinear activations between hidden layers, stacking affine transformations still gives an affine transformation. **Backpropagation** computes gradients efficiently; an optimizer uses them to update parameters. Hidden units compute learned activations, rather than being unexplained “unknown variables.”

Deep networks can learn useful features from images, audio, and text. The PDF's driving example illustrates representation learning: road-related image features can emerge from a steering prediction objective. Use that to explain the idea without implying that every hidden unit has a neat human-readable meaning or that deep learning is always the best choice. Neural architectures can also be trained with unsupervised, semi-supervised, or reinforcement signals.

### Extra names from the PDF: recognition only

| Name | Enough for this overview |
| --- | --- |
| Quadratic classifiers | Permit quadratic decision boundaries. [QDA](https://scikit-learn.org/stable/modules/lda_qda.html) models class-conditional Gaussian distributions with separate covariance matrices, which can yield such boundaries. |
| Least-squares SVM | A related SVM formulation with squared errors and equality constraints, commonly expressed through linear equations; see the [KU Leuven introduction](https://www.esat.kuleuven.be/sista/statlssvm/). |
| Kernel estimation | A broader family of methods that use local similarity to estimate quantities such as density. KNN is also neighborhood-based, but is not interchangeable with every kernel method. |
| Learning vector quantization | A supervised nearest-prototype approach. Labeled representatives are adjusted so their positions support classification; basic forms use a winner-take-all nearest-prototype step. |

Do not infer that all Naive Bayes variants always have linear boundaries, or that every kernel-based method is an SVM. Groupings in an introductory algorithm list are helpful signposts, not strict equivalences.

## 7. Clustering: groups depend on the representation

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

## 8. Answers and closing discussion

The three HTML checks should produce these explanations:

1. **Regression:** the target is minutes, and the examples include observed times. Changing the target to “late / on time” changes this to classification.
2. **Validation:** choose K using validation or cross-validation. Leave the test set for the final chosen procedure.
3. **Overfitting is a reasonable first suspicion:** the model handles its training examples better than held-out examples. Inspect complexity, leakage, the split, and whether the data distribution has changed.

A final verbal prompt: “We have delivery records but no late labels. Can K-means discover the late class?” It can form groups from available features, but it has no evidence that a particular group means “late.” A cluster needs interpretation and external validation before receiving that meaning.

Students should be able to leave with four questions: **What are we predicting? What information is available? How does the model learn? How do we know it generalizes?**

## Source coverage and presentation choices

The lesson synthesizes these supplied local materials:

- [Lab 06 HTML](../../AI_FMI_course/docs/labs/lab06-intro-ml.html): core terminology, bias and variance, training vocabulary, regression, model families, and K-means.
- [Lab 06 notebook](../../AI_FMI_course/notebooks/06_Introduction_to_ML.ipynb): the same main learning objectives and algorithm overview.
- [Intro ML slides, Boris Velichkov](Intro%20ML.pdf): induction versus deduction; semi-supervised and reinforcement learning; the AI / ML / NN / DL relationship; preprocessing, transformation and reduction; additional classifier names; train / validation / test and cross-validation; hierarchical clustering and linkage.

The wording and diagrams are new. The lesson preserves the concepts rather than repeating unsupported popularity claims, overly broad statements, or source-specific examples. The many preprocessing options and additional classifier variants receive a short mention in the HTML and fuller context here, keeping the visible lesson teachable in 90 minutes.

All interactive data is synthetic. The visualizations use inline SVG and plain JavaScript, with fixed seeds where randomness is involved; no external assets or services are needed. The page can be opened directly from disk. Use **Hide route** in the header to hide the learning route and give the lesson the full available width; **Show route** restores it. The browser remembers this preference when local storage is available.
