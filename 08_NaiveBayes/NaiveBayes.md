# Naive Bayes Classifier

## Deriving the Naive Bayes model

### 1. Bayes’ formula with one attribute

$$
\Large P(c\mid x)=\frac{P(c)\,P(x\mid c)}{P(x)}
$$

Here, $c$ is the class we want to predict and $x$ is the observed input. Bayes’ formula updates the probability of a class after we observe new information.

#### Example

| Outlook | Play = Yes | Play = No | Total |
|---|---:|---:|---:|
| Sunny | 2 | 3 | 5 |
| Overcast | 4 | 0 | 4 |
| Rain | 3 | 2 | 5 |
| **Total** | **9** | **5** | **14** |

We observe $x=\text{Sunny}$ and want $P(\text{Yes}\mid\text{Sunny})$. Read the three ingredients from the table:

* **Prior:** $P(\text{Yes})=9/14$. We played on 9 of all 14 days, so our estimate before knowing the outlook is about **64%**.
* **Likelihood:** $P(\text{Sunny}\mid\text{Yes})=2/9$. Among the 9 days when we played, 2 were sunny.
* **Evidence:** $P(\text{Sunny})=5/14$. Of all 14 days, 5 were sunny.

Substitute these values into Bayes’ formula:

$$
P(\text{Yes}\mid\text{Sunny})
=\frac{(9/14)(2/9)}{5/14}
=\frac{2/14}{5/14}
=\frac25=0.40.
$$

The **posterior** is **40%**: among the 5 sunny days, we played on 2. Knowing the outlook changes our estimate from about **64% to 40%**.

### 2. Adding a second attribute

Suppose today is **sunny with high humidity**. We now need the probability of seeing both conditions within each class.

| Class | Days | Sunny | High humidity | Both sunny and high |
|---|---:|---:|---:|---:|
| Yes | 9 | 2 | 3 | 0 |
| No | 5 | 3 | 4 | 3 |

**Without an independence assumption**, the multiplication rule gives

$$P(\text{Sunny, High}\mid c)=P(\text{Sunny}\mid c)\,P(\text{High}\mid\text{Sunny},c).$$

For the No class, 3 of the 5 days were sunny. All 3 of those sunny days had high humidity. The observed joint frequency is therefore $(3/5)(3/3)=3/5$.

**With conditional independence**, we estimate humidity using all the days in the class. Naive Bayes uses

$$P_{\mathrm{NB}}(\text{Sunny, High}\mid c)=P(\text{Sunny}\mid c)\,P(\text{High}\mid c).$$

| Class | Observed joint frequency | Naive Bayes estimate |
|---|---:|---:|
| Yes | $0/9=0$ | $(2/9)(3/9)\approx0.074$ |
| No | $3/5=0.60$ | $(3/5)(4/5)=0.48$ |

* **What we gain.** We can score a combination even when it never appeared in that class, provided its individual values appeared. With four weather attributes, there are 36 possible combinations and only 14 training days.
* **What we give up.** Within the No class, high humidity occurred on all 3 sunny days and on 4 of the 5 days overall. The model uses $4/5$ in both situations. Its estimate can be off while still ranking the classes correctly.

### 3. The classification rule

For a new example with $d$ attributes, combine the prior with one likelihood per observed attribute. The denominator $P(x)$ is the same for every class, so comparing the numerators gives the prediction.

$$s(c)=P(c)\prod_{j=1}^{d}P(x_j\mid c),\qquad \hat c=\arg\max_c s(c).$$

**Full example from the lecture.** The new day is **sunny, cool, high humidity, strong wind**. Each fraction below comes from counting days within the corresponding class.

| Class | Prior | Sunny | Cool | High humidity | Strong wind |
|---|---:|---:|---:|---:|---:|
| Yes | $9/14$ | $2/9$ | $3/9$ | $3/9$ | $3/9$ |
| No | $5/14$ | $3/5$ | $1/5$ | $4/5$ | $3/5$ |

$$s(\text{Yes})=\frac9{14}\cdot\frac29\cdot\frac39\cdot\frac39\cdot\frac39\approx0.00529$$

$$s(\text{No})=\frac5{14}\cdot\frac35\cdot\frac15\cdot\frac45\cdot\frac35\approx0.02057$$

* **Prediction.** The larger score is for **No**. Cool weather favors Yes, while sunshine, high humidity and strong wind each favor No.
* **Probability.** Divide the winning score by the sum of both scores. The model assigns No a probability of $0.02057/(0.02057+0.00529)\approx79.5\%$.

## Fitting the model

1. **Estimate the priors.** Count the examples in each class and divide by the total number of examples. In the weather data, $\widehat P(\text{Yes})=9/14$.
2. **Estimate the attribute distributions.** For each class, fit one distribution per attribute. For outlook on Yes days, the probabilities of sunny, overcast and rain are $2/9$, $4/9$ and $3/9$.
3. **Connect the counts to maximum likelihood.** Let $p$ be the chance of sunshine on a Yes day. We observed 2 sunny and 7 other days. Assuming independent training days, the likelihood of this sequence is $p^2(1-p)^7$. Its maximum occurs at $p=2/9$. The relative frequency we counted is the maximum likelihood estimate.

**Number of parameters.** A binary attribute needs one probability per class because the other is its complement. A three category attribute needs two. For the weather data, outlook, temperature, humidity and wind need $2+2+1+1=6$ free probabilities per class. With two classes and one free class prior, the model has **13 free parameters**.

## Handling different types of attributes

### Categorical attributes and discrete ratings

* **Categorical.** Fit a probability for each category within each class. Outlook can be stored as sunny, overcast or rain. A categorical model treats these as distinct labels, so one hot encoding is unnecessary.
* **Discrete numerical.** A rating from 1 to 5 can use the same approach. If 12 of 30 customers who renewed gave a rating of 4, then $\widehat P(\text{rating}=4\mid\text{renewed})=12/30$. Each rating has its own count. A rating of 5 does not contribute to the count for 4.

### Continuous attributes with a Gaussian distribution

* Fit a separate mean $\mu_c$ and standard deviation $\sigma_c$ for the attribute within each class. Use the height of that class’s Gaussian curve at the observed value as the likelihood factor.
* **Example from the lecture.** Consider only weight from the height, weight and foot size example. The observed weight is 140 lb. The fitted means are 176.25 and 132.50 lb for the two groups.

| Group in the lecture | Mean weight | Standard deviation | Density at 140 lb |
|---|---:|---:|---:|
| Male | 176.25 lb | 11.09 lb | 0.000172 per lb |
| Female | 132.50 lb | 23.63 lb | 0.0161 per lb |

* With equal priors, weight alone favors the second group by about $0.0161/0.000172\approx94$ to 1. To use all three measurements, multiply the prior by the height, weight and foot size densities within each group.
* A density describes how concentrated observations are near 140 lb. The probability of falling in an interval, such as 139 to 141 lb, is the area under the curve across that interval.

### Continuous attributes with bins

Suppose a machine runs in two modes. Half its jobs finish near 100 ms and half near 180 ms. A single Gaussian would peak near the mean of 140 ms, where few jobs actually finish.

| Run time | Jobs on this machine | Estimated probability |
|---|---:|---:|
| Below 120 ms | 49 | 0.49 |
| 120 ms up to 160 ms | 2 | 0.02 |
| 160 ms or more | 49 | 0.49 |

* A new reading of 143.7 ms uses the middle bin and gets a factor of $0.02$. The exact reading can be new. We only need its interval.
* Use the same boundaries across classes. The outer bins cover future values beyond the training range. If a bin is empty in one class, smoothing can give it a small positive probability.
* Bins can represent multiple peaks. Narrower bins give more detail but leave fewer observations per bin.

## Log scores, smoothing and validation

### Log scores for small products

Multiplying 1,000 factors of $0.1$ gives $10^{-1000}$, which rounds to zero in ordinary double precision. We can add the logarithms of the factors directly.

$$\log s(c)=\log P(c)+\sum_{j=1}^{d}\log P(x_j\mid c).$$

The logarithm is increasing, so the class with the highest score also has the highest log score.

### Laplace smoothing for zero counts

None of the 5 No days was overcast. With raw counts, the factor $0/5$ makes the whole score zero. In log space it contributes $-\infty$.

* Add $\alpha$ to the count for **every** possible category. For $V$ categories and $N_c$ examples in class $c$, the denominator increases by $\alpha V$.

$$\Large \widehat P(X_j=v c)=\frac{N_{c,v}+\alpha}{N_c+\alpha V}.$$

* **Laplace smoothing** uses $\alpha=1$. Here the three outlook counts change from $(3,0,2)$ to $(4,1,3)$. The smoothed probabilities are $4/8$, $1/8$ and $3/8$. An overcast outlook now contributes $1/8$ to the No score, so the other attributes still influence the prediction.

### Cross validation

* Compare feature sets, bin choices and values of $\alpha$ on the training data using the same validation folds.
* Fit feature selection, bin boundaries and model parameters using only the training portion of each fold. Refit the selected configuration on all training data, then evaluate it on the test set.

## Advantages and disadvantages

| Advantages | Disadvantages |
|---|---|
| Training and prediction are fast. Each attribute is processed separately. | The model can miss relationships that matter only when attributes are considered together. |
| Few parameters make it practical with limited data and many attributes. | Rare categories and narrow bins can still have unreliable estimates. |
| The model is simple to implement. We can inspect which factors favored each class. | Related attributes can count the same evidence twice and produce overconfident probabilities. |

**Example of duplicate evidence.** Suppose *free* appears in 30% of spam and 5% of ordinary email. With equal priors, observing *free* gives spam odds of $0.30/0.05=6$ to 1. If we copy the same indicator into a second column, the model multiplies by 6 again and reports odds of 36 to 1. Keep one copy. For related attributes that carry some distinct information, compare feature sets using validation.

## Practical use cases

* **Spam detection.** In the notebook, 40% of emails are spam and 60% are ordinary. Use two binary features, whether *free* and *money* appear. For an email containing both, the spam score is $0.40\times0.30\times0.25=0.030$. The ordinary email score is $0.60\times0.05\times0.10=0.003$. The prediction is **spam**.
* **Weather forecasting.** To predict rain tomorrow, label each training day by whether it rained the following day. Use observations available today, such as pressure and humidity. Validate on later days to reflect how the forecast will be used.
* **Customer renewal.** Predict whether a customer will renew using plan type, satisfaction rating and usage. Estimate plan and rating probabilities from counts. Model usage with a Gaussian or bins depending on its distribution within each class.
