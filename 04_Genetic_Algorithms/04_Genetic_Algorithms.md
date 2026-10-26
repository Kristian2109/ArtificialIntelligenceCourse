# Genetic Algorithms

**The central question:** Can a population of imperfect answers produce better answers over several generations?

**Content:**
- Chromosomes, representations, and fitness
- Parent selection and selection pressure
- Crossover, mutation, and elitism
- A complete genetic algorithm
- Travelling salesman and N-queens examples
- Knapsack homework

Run the cells from top to bottom using Python 3. The examples use only the standard library and fixed random seeds.

```python
import random
from math import ceil, dist
```

## Concepts
> Goal: Maintain several candidate solutions, then use them to generate a new population.

- **gene**: one component of a solution.
- **chromosome / individual**: one complete solution.
- **population**: set of solutions which is in one iteration.
- **generation**: one iteration of replacing the current population with a new population.
- **fitness score**: evaluation of a given solution

## Example
### Problem

Minimize the **Rosenbrock function** for $n\geq 2$:

$$
\min_{\mathbf{x}\in[-3,3]^n} f(\mathbf{x})
=\sum_{j=1}^{n-1}\left[100(x_{j+1}-x_j^2)^2+(1-x_j)^2\right].
$$

Neighboring variables interact, making the narrow, curved valley toward the optimum $\mathbf{x}^*=[1,\ldots,1]$ difficult to navigate.

### Solution → gene → population

- **Chromosome / individual:** one complete candidate solution, $\mathbf{x}=[x_1,\ldots,x_n]$, e.g. $[0.3,1.8,...,-0.2]$.
- **Gene:** one number $x_j$ in that solution, e.g. the second gene is $1.8$.
- **Population:** $m$ candidate solutions, $P=[\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}]$, each containing $n$ genes.
- **Fitness score**: could be the value of the function given a particular chormosome. The closer to the 0, the better is the chromosome.

## The generation cycle

```text
Initialize → Evaluate → Selection → Crossover → Mutate → New Population → Conversion Check → Final solution
                ↑                                                                │
                └────────────────────────────────────────────────────────────────┘
```

- Initialization: Create a starting population
- Evaluate: Compute the fitness scores of the population chromosomes
- Selection: Select which chromosomes to continue the population
- Crossover: Combine the selected solutions in order to produce new
- Mutation: Mutate the surviving solutions in order to increase the diversity
- Conversion Check: Check if the population continues improving or has potential to improve
  - If it has not converged, start a new iteration
  - If it has converged, we've reached the final solution

## Selection

### Selection pressure

> Control how strongly parent selection favors better individuals.

- Strong pressure spreads good traits quickly, but may remove useful diversity too early.
- Weak pressure gives weaker individuals more opportunities, but may lead to slow improvement.
- Selecting a parent does not guarantee that its child is better.

```python
population = ['A', 'B', 'C', 'D', 'E']
scores = [10, 20, 30, 25, 15]
```  


### Roulette-wheel selection

> Give each individual a share of the selection probability proportional to its fitness.

$$P(i)=\frac{f_i}{\sum_j f_j}$$

- Sampling uses replacement: a parent may be chosen more than once.
- Large score differences can make one individual dominate.
- Usable for nonnegative fitness scores  

```python
def roulette_select(population, scores, rng, count=1):
    return rng.choices(population, weights=scores, k=count)  
```


### Tournament selection

> Sample k indviduals on random from the population. From them select the best candidate.  

- $k$ contolls the selection pressure.
  - Higher $k$ leads to a higher selection pressure
  - $k=1$ makes the choice of the candidate random

```python
def tournament_select(population, scores, rng, count=1, size=3):
    selected = []
    for _ in range(count):
        candidates = rng.sample(range(len(population)), size)
        winner = max(candidates, key=lambda i: scores[i])
        selected.append(population[winner])
    return selected

print('Tournament:', tournament_select(population, scores, random.Random(7), count=5))
```


### Rank-based selection

> Sort the individuals by fitness. Give better-ranked individuals a higher probability of being selected.

- Selection depends on rank, so large score differences do not make one individual dominate.
- Equal scores receive the same rank in this implementation.
- Sampling uses replacement: a parent may be chosen more than once.

```python
def rank_select(population, scores, rng, count=1):
    ranks = {score: rank for rank, score in enumerate(sorted(set(scores)), start=1)}
    weights = [ranks[score] for score in scores]
    return rng.choices(population, weights=weights, k=count)
```

### Stochastic universal sampling

![alt text](images/SUS.png)

> Give each individual a segment proportional to its fitness, as in roulette-wheel selection. Select parents using equally spaced pointers with one random starting point.

- The distance between pointers is total fitness divided by the number of parents to select.
- Selects parents more evenly according to their fitness than independent roulette-wheel draws.
- A parent may be chosen more than once.
- Usable for nonnegative fitness scores. All-zero scores receive equal weights below.

```python
def sus_select(population, scores, rng, count=1):
    weights = list(scores)
    spacing = sum(weights) / count
    start = rng.random() * spacing
    selected, index, cumulative = [], 0, weights[0]

    for i in range(count):
        pointer = start + i * spacing
        while index < len(population) - 1 and pointer >= cumulative:
            index += 1
            cumulative += weights[index]
        selected.append(population[index])
    return selected
```

### Elitism

> Copy a small number of the best current individuals directly into the next generation.

- Elites survive unchanged; they do not go through crossover or mutation.
- Keeping the best individual preserves the best score when the fitness function stays the same.
- The number of elites controls how much of the population is preserved.
  - More elites preserve more existing solutions but leave less room for new candidates.
  - Too many elites can reduce diversity and slow improvement.

```python
def keep_elites(population, scores, count=1):
    order = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    return [population[i].copy() for i in order[:count]]
```

### Truncation selection

> Sort the individuals by fitness. Keep only the top fraction, then select parents randomly from that group.

- The retained fraction controls the selection pressure.
  - A smaller fraction leads to higher selection pressure and can reduce diversity.
  - A fraction of 1 makes parent selection random across the whole population.
- All retained individuals have the same probability of being selected.
- Sampling uses replacement: a parent may be chosen more than once.
- The opposite of the tournament selection

```python
def truncation_select(population, scores, rng, count=1, fraction=0.4):
    order = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    retained = [population[i] for i in order[:ceil(fraction * len(population))]]
    return rng.choices(retained, k=count)
```

### Compare selection policies

| Policy | What controls selection? | Main tradeoff |
|:--|:--|:--|
| Roulette | Relative fitness weights | Sensitive to score differences |
| Tournament | Best of $k$ sampled individuals | Tournament size controls pressure |
| Rank-based | Ordered fitness levels | Ignores the size of fitness gaps |
| SUS | Equally spaced weighted pointers | Lower sampling variance |
| Elitism | Best individuals survive unchanged | Protects progress but reduces turnover |
| Truncation | Membership in the top fraction | Strong pressure can remove diversity |

## Crossover

> Combine genes from two parents to create a child.

- The crossover method must match the chromosome representation.
- Combining good parents does not guarantee a better child.

### One-point crossover

> Choose one random cut. Take the prefix from one parent and the suffix from the other.

- Suitable for bit strings and other sequences where mixing genes keeps the solution valid.
- Both parents must have the same length.

### Multi-point crossover

- Suitable for equal-length bit strings and other sequences where mixing genes keeps the solution valid. Permutations require a repair step if duplicates occur.

### Uniform crossover

> Independently choose each gene from either parent, usually with equal probability.

- Mixes genes more freely than one-point crossover.
- Can disrupt groups of genes that work well together.


### Order crossover

> Keep a segment from one parent, then fill the remaining positions in the order they appear in the other parent, skipping duplicates.

- Preserves each gene exactly once.
- Both parents must contain the same unique genes.

### Partially mapped crossover (PMX)

- Suitable for permutation problems, such as tours and task ordering, where each gene must appear exactly once. Both parents must contain the same unique genes.

### Other crossover choices

| Method | How it combines parents | Suitable representation |
|:--|:--|:--|
| Arithmetic | Blend corresponding values using a weighted average | Real-valued vectors |
| Cycle | Copy groups of positions linked by cycles between parents | Permutations |

### Compare crossover methods

The use cases below are guidance inferred from what each operator preserves, rather than a ranking of performance.

| Method | Suitable representation | Appropriate when | What it preserves | Main tradeoff |
|:--|:--|:--|:--|:--|
| [One-point][crossover-bias] | Bit strings; fixed-length value sequences | Useful genes are grouped into contiguous blocks | A prefix and a suffix at their original positions | Can split useful blocks; inheritance depends on gene placement |
| [Multi-point (including two-point)][crossover-bias] | Bit strings; fixed-length value sequences | Several parental blocks should contribute to a child | Contiguous segments at their original positions | More cuts allow more mixing but can break more useful groups |
| [Uniform][crossover-bias] | Bit strings; fixed-length value sequences | Genes should mix independently of their distance along the chromosome | Each value comes from a parent at the same position | Can disrupt combinations of genes that work well together |
| [Order (OX1)][permutation-crossover] | Permutations of the same unique genes | Relative order matters, as in task ordering or tour representations | One parent's segment and the other parent's cyclic order of remaining genes | Remaining genes can change positions; parental tour edges are not guaranteed |
| [Partially mapped (PMX)][permutation-crossover] | Permutations of the same unique genes | Retaining a parental segment at fixed positions is useful | A segment at its original positions, plus other parental placements where conflicts allow | Mapping can change outside placements; relative order is not guaranteed |
| [Cycle (CX)][permutation-crossover] | Permutations of the same unique genes | Gene-to-position assignments matter | Every gene occupies the position it had in one of the parents | Can reproduce a parent unchanged; does not specifically preserve order or tour edges |
| [Arithmetic][arithmetic-crossover] | Real-valued vectors | Intermediate parameter values are meaningful | With weights in $[0,1]$, each blended value lies between the corresponding parental values | Repeated averaging can reduce diversity; unsuitable for categorical labels or permutations |

## Mutation

> Randomly change genes in a chromosome to introduce variation.

- Helps maintain diversity and explore new solutions.
- The mutation method must match the chromosome representation.
- The mutation rate controls how often changes occur.
  - Too little mutation can leave the population stagnant.
  - Too much mutation can destroy useful combinations of genes.

### Bit-flip mutation

> Independently flip each bit with a small probability.

- Suitable for binary chromosomes: a selected bit changes from 0 to 1 or from 1 to 0.
- The rate applies independently to each bit.
  - With rate $p$ and chromosome length $L$, the expected number of flips is $Lp$.
  - One mutation step may flip no bits, one bit, or several bits.

### Swap mutation

> Exchange two positions without adding or losing genes.

- Suitable for tours and other permutations: each gene still appears exactly once.
- Choose two different positions randomly and exchange their genes.
- The rate is the probability of making one swap per individual.
  - A mutation step makes either no swap or one swap.


### Insertion mutation

- Randomly select a gene, remove it, and insert it at a different position.
- Suitable for permutations because each gene still appears exactly once.

### Inversion mutation

> Reverse the order of genes within a selected segment.

- Choose two different positions randomly and reverse the segment between them, including both endpoints.
- Suitable for tours and other permutations: each gene still appears exactly once.
- Example: reversing positions 2–4 changes `[A, B, C, D, E]` to `[A, D, C, B, E]`.

## Considerations

- **Time complexity**: typically $O(G \cdot P \cdot (F + L))$ for $G$ generations, population size $P$, fitness evaluation cost $F$, and chromosome length $L$. Selection or sorting may add extra cost.
- **Space complexity**: typically $O(P \cdot L)$ to store the populations, plus any recorded history.
- **Guarantees**: no guarantees for a valid solution or the global optimum.
- **Stopping**: use a generation limit, target fitness, or lack of improvement.
- **Problem specific**: always select methods according to the problem in front and **combine different methods**
  - **Elitism** is very often combined with another selection technique because it preserves the best inviduals so degradation is prevented while using the other technique to increase the variability

## Travelling salesman problem

> Visit every city once and return to the starting city with minimum total distance.

- **Chromosome**: a permutation of city labels.
- **Objective**: sum consecutive distances, including the closing edge to the first city.
- **Fitness**: $1/(1+distance)$; shorter tours receive larger positive scores.
- Order crossover and swap mutation preserve valid tours. The starting city can change without changing the underlying cycle.

## N-queens with a population

> Evolve queen placements instead of repairing just one board as in min-conflicts.

- Gene $c$ stores the queen's row in column $c$.
- Initialize a permutation of rows: each row and column already contains exactly one queen.
- **Fitness**: $n(n-1)/2$ minus the number of attacking pairs. The maximum means no attacks.
- Use order crossover and swap mutation to retain row uniqueness.
- This transparent fitness function checks pairs in $O(n^2)$. Row/diagonal counters from the CSP lecture can reduce evaluation work.

```text
Queen rows by column: [6, 3, 1, 7, 5, 0, 2, 4]
Attacking pairs: 0
Generations completed: 3
. . . . . Q . .
. . Q . . . . .
. . . . . . Q .
. Q . . . . . .
. . . . . . . Q
. . . . Q . . .
Q . . . . . . .
. . . Q . . . .
```

## Knapsack problem

> Choose valuable items without exceeding the available weight capacity.

- **Chromosome**: one bit per item; 1 includes the item, 0 excludes it.
- **Hard constraint**: total weight must not exceed capacity.
- Below, an overweight chromosome receives fitness 0; a feasible chromosome receives its total value.
- A penalty can leave many invalid individuals tied at zero. Alternatives include repairing overweight chromosomes or prioritizing feasibility explicitly.
- Crossover and bit mutation preserve the encoding, but not necessarily the capacity constraint.
