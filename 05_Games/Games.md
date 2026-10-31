
## 1. Games

### What is a game?

A game is a decision problem with **multiple players whose actions affect each other's outcomes**.

A game model specifies an **initial state**, the **player to move**, **legal actions**, a **transition model**, **terminal states**, and **utilities (payoffs)**.

### Perfect information

**Every player observes the full game state and all previous moves.** There are no hidden cards, pieces, or actions. Example: chess.

### Complete information

**The rules, available strategies, and every player's payoff function are common knowledge.** There are no private payoff types or incentives.

Perfect information concerns **what players observe**; complete information concerns **what players know about the game and each other's payoffs**.

### Deterministic vs. stochastic games

- **Deterministic:** the current state and chosen action determine the next state.
- **Stochastic (nondeterministic):** chance also affects the next state, such as a dice roll or shuffled cards.

### Examples

| Game | Perfect information? | Complete information? | Chance |
|:--|:--|:--|:--|
| Tic-Tac-Toe, chess, Go | Yes | Yes | No |
| Backgammon | Yes; dice outcomes are visible | Yes | Yes: dice rolls |
| Battleship | No: hidden ship positions | Yes | No, after setup |
| Poker (known monetary payoffs) | No: hidden cards | Yes | Yes: shuffled deck |
| Sealed-bid auction with private valuations | No: hidden bids | **No: private item values determine bidders' payoffs** | No, for fixed valuations |

**Hidden state does not necessarily mean incomplete information:** players can know the rules and payoff functions without seeing each other's cards.

**Incomplete-information example:** in the auction, a bidder's payoff from winning is their private item value minus the price paid. Other bidders know the rules but not that value.

## 2. Minimax

$
V(s)=
\begin{cases}
U(s), & \text{if }s\text{ is terminal}\\[4pt]
\max\limits_{s'\in Succ(s)}V(s'), & \text{MAX's turn}\\[6pt]
\min\limits_{s'\in Succ(s)}V(s'), & \text{MIN's turn}
\end{cases}
$

### Idea

> **Choose the move with the best guaranteed outcome, assuming the opponent plays optimally.**

- **MAX** maximizes the score; **MIN** minimizes it. All scores use **MAX's perspective**.
- In the **game tree**, nodes are states and edges are legal moves.
- Search to terminal states, evaluate them, then propagate values upward: **maximum at MAX nodes, minimum at MIN nodes**.
- At the root, choose the move leading to the best child value.

For Tic-Tac-Toe, with X as MAX: **X wins → +1**, **O wins → −1**, **draw → 0**.

### Conditions

Standard Minimax assumes:

- **Two players taking alternating turns.**
- **Zero-sum payoffs:** one player's gain is the other's loss.
- **Perfect and complete information:** the state, legal moves, and payoffs are known.
- **Deterministic transitions:** no chance nodes.
- **Full search to terminal states** for an exact optimality guarantee.

A **depth limit** replaces terminal utilities with a **heuristic evaluation** at the cutoff; the chosen move then depends on the heuristic's accuracy. Chance requires expected-value nodes, while hidden information requires reasoning over possible states.

### Algorithm

The following Python example uses nested lists for internal nodes and integers for terminal utilities. Each internal node must have at least one child.

```python
def is_terminal(node):
    return isinstance(node, int)


def evaluate(node):
    return node


def get_children(node):
    return node


def minimax(node, maximizing_player):
    if is_terminal(node):
        return evaluate(node)

    if maximizing_player:
        best = float("-inf")
        for child in get_children(node):
            best = max(best, minimax(child, False))
        return best

    best = float("inf")
    for child in get_children(node):
        best = min(best, minimax(child, True))
    return best
```

**This function returns a value.** To select a move, evaluate each root child and choose the highest value for MAX or the lowest for MIN. For Tic-Tac-Toe, replace the three helper functions with board-specific logic.

### Properties

Let **$b$** be the maximum branching factor and **$m$** the maximum search depth.

| Property | Minimax |
|:--|:--|
| **Completeness** | Yes, for a finite game tree. Cycles require termination rules or repetition handling. |
| **Optimality** | Guarantees the best worst-case outcome under the conditions above; it does **not** guarantee a win. |
| **Time complexity** | $O(b^m)$: explores the full tree. |
| **Space complexity** | $O(bm)$ with depth-first search storing each node's children; $O(m)$ if successors are generated one at a time. Excludes any prebuilt input tree. |

## 3. Alpha-Beta pruning

### Idea

> **Skip branches that cannot improve the decision.** Alpha-Beta computes the same Minimax value while exploring fewer nodes.

- **$\alpha$:** the best (highest) value MAX can already guarantee along the current search path; initially $-\infty$.
- **$\beta$:** the best (lowest) value MIN can already guarantee along the current search path; initially $+\infty$.
- At MAX nodes, raise $\alpha$ when a better value is found; at MIN nodes, lower $\beta$.
- **Prune when $\alpha \geq \beta$:** an earlier alternative is already at least as good, so the remaining children cannot affect the decision.

### Properties

| Property | Alpha-Beta pruning |
|:--|:--|
| **Result** | Same root value as Minimax; may choose a different move when moves tie. |
| **Completeness and optimality** | Same guarantees and conditions as Minimax. |
| **Worst-case time** | $O(b^m)$ with poor move ordering. |
| **Best-case time** | $O(b^{m/2})$ with ideal move ordering. |
| **Space** | Same depth-first space requirements as Minimax. |

**Move ordering matters:** try promising moves first—high-value moves for MAX, low-value moves for MIN. Better ordering produces more pruning.

### Visualization

**In the separate Tic-Tac-Toe visualization:** display $\alpha$ and $\beta$, mark pruned branches, and compare visited-node counts with plain Minimax. The root value must match.

## 4. Homework

### Tic-Tac-Toe with Minimax and Alpha-Beta pruning

1. Represent the board, legal moves, player turns, and terminal states; reject invalid moves.
2. Implement **Minimax** using **+1 / −1 / 0** terminal utilities from X's perspective.
3. Implement **Alpha-Beta pruning** with the same utilities and move order.
4. Allow a human to play either X or O against either algorithm.
5. Compare **chosen moves, root values, visited nodes, and execution time** on the same positions. Values must match; tied moves may differ.

**Expected behavior:** optimal play from both sides produces a draw, and the AI never loses from the empty board. Alpha-Beta visits no more nodes than Minimax with the same move order.
