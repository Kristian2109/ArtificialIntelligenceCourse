Presentation plan

1. Review of the Bayesian formula
- Show the Bayes formula
2. Relating to the idea for indenpendence of the input factors
- Retrieving the conditional probability with and without independence
- Providing an example to showcase the simplification
- The independence assumption is almost always not true, but very often 
3. Retrieving the Naive Bayes formula by also removing the redundant weights and getting the main idea
- The naive bayes formula is a combination between how probable is the given class as a whole and how probable are the input artributes for the given class
- The fitting is when we count the conditional probabilities for each class, the total number of parameters is number_of_classes * number_attributes
4. Relate the practical and simple explanation to the statistical learning theory in simple terms
- We can show the main idea that we have a distribution and we want to fit the exampels to the distribution extracting the different parameters using maximum likelihood estimation.
5. Different kinds of attributes
- Categorical: handles them very well, no need for one hot encoding as it doesn't assume an order between the categories
- Discrete numerical, like rating, same as the categories.
- Continuous (Gaussian): assume that the data follows the Gaussian distribution and based on this compute the probability density, give a concrete example. Provide an example.
- Continuous discretization: use the discretized density (actually we always use the density), if the value is not found, the Laplace Smoothing would just give the minimalistic value. This is appropriate when the feature follows a specific distribution where the mean and variance doesn't bring enough information like a distribution with multiple heads. Provide an example.
6. Optimizations
- Logarithmic scalling: when we have multiple values, the result is getting so small, that it can go below the precision supported by the computer. The logarithm preserves the increasing and decreasing, but converts the multiplication into sums
- Laplace Smoothing: if a given class does not have a specific feature, it's density there is 0, which will nullify the whole product, or in case of logarithm 0 is minus infinity or not defined, so this doesn't do the job. Explain the idea, giving a simple formula and providing the rationale
7. Advantages and disadvantages
Create a table with them
Advantages
- Fast and efficient
- Relatively good for interpretation
- Does not need too much data
- Simple to implement and understand 
- Works well with high dimensional data
Disadvantages
- "Naive", the independence assumption could lead to deeply flawed results if there is important relationship between the attributes
- Inappropriate features can worsen the results. Give an example with feature duplication or very correlated features (in that case the high correlation leads to increasing the factor of this dimension), so the input features should be properly selected and if necessary compressed
8. Practical Use Cases
- Weather forecast: just a good example
- Spam detection
- Another example just to showcase this method