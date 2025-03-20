# 1. Introduction to Dynamic Programming

**Dynamic Programming (DP)** is a problem-solving technique that involves breaking a complex problem into simpler overlapping subproblems and solving each subproblem only once. In other words, DP trades redundant computation for memory by storing solutions to subproblems and reusing them when needed ([Overlapping Subproblems Property in Dynamic Programming | DP-1 - GeeksforGeeks](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/#:~:text=Dynamic%20Programming%20is%20an%20algorithmic,be%20solved%20using%20Dynamic%20programming)). A problem that can be solved with DP typically has two key properties: **overlapping subproblems** and **optimal substructure** ([Overlapping Subproblems Property in Dynamic Programming | DP-1 - GeeksforGeeks](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/#:~:text=Dynamic%20Programming%20is%20an%20algorithmic,be%20solved%20using%20Dynamic%20programming)) ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=A%20given%20problem%20is%20said,way%20to%20solve%20the%20subproblems)).

- *Overlapping subproblems* means the problem’s recursive subproblems are reused multiple times. A classic example is computing Fibonacci numbers recursively: the naive recursion computes the same subvalues repeatedly (e.g. F(2) is recalculated many times) ([A graphical introduction to dynamic programming | by Avik Das | Medium](https://medium.com/@avik.das/a-graphical-introduction-to-dynamic-programming-2e981fa7ca2#:~:text=The%20fundamental%20issue%20here%20is,time%20that%E2%80%99s%20exponential%20in%20n)). DP avoids this by storing results. If a problem has no overlapping subproblems (e.g. binary search), DP won’t be beneficial ([Overlapping Subproblems Property in Dynamic Programming | DP-1 - GeeksforGeeks](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/#:~:text=,are%20solved%20again%20and%20again)).
- *Optimal substructure* means an optimal solution can be constructed from optimal solutions of its subproblems ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=A%20given%20problem%20is%20said,way%20to%20solve%20the%20subproblems)). For instance, the shortest path problem has optimal substructure: the shortest path from A to C via B is the shortest path from A to B plus the shortest path from B to C ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=Example%3A%C2%A0The%20Shortest%20Path%20problem%20has,the%20following%20optimal%20substructure%20property)). If a problem lacks this property (e.g. finding the longest path in a graph can fail this), then DP may not yield an optimal solution.

When these properties hold, we can use DP to solve the problem efficiently by combining solutions of subproblems. DP solutions involve three main steps: **define the state**, **formulate the recurrence**, and **identify base cases**. The *state* represents a configuration of the problem defined by a few parameters (ideally as few as possible) that uniquely identify a subproblem ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=State%3A)). For example, in a knapsack problem the state might be defined by two parameters: the index of the current item and the remaining capacity ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=Example%3A%20Let%E2%80%99s%20take%20the%20classic,subproblem%20we%20need%20to%20solve)). Once the state is defined, we **formulate a recurrence** to express the solution of a state in terms of solutions of smaller states (for knapsack, the choice to include or exclude an item). Finally, we set up the **base cases** for the smallest subproblems (e.g. 0 items or 0 capacity).

**DP vs Recursion:** A plain recursive or brute-force solution recomputes subproblems, leading to exponential time in many cases ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=Dynamic%20Programming%20is%20mainly%20an,time%20complexity%20and%20if%20we)). DP optimizes recursion by storing intermediate results (memoization) or building solutions bottom-up (tabulation), reducing time complexity from exponential to polynomial in typical scenarios ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)).

**Top-Down (Memoization) vs Bottom-Up (Tabulation):** These are two approaches to implementing DP ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=Programming,implemented%20by%20memoization%20or%20tabulation)) ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=%2A%20Top,Entries%20are%20filled%20when%20needed)):

- **Memoization (Top-Down):** Solve the problem recursively and cache the results of subproblems in a lookup table (e.g. dictionary). Before computing a subproblem, check the cache; if it exists, reuse it instead of recomputing. This approach is top-down because it starts from the original problem and recursively breaks it down, storing results as it returns. It’s often easier to implement by adding a cache to a recursive solution. For example, computing Fibonacci with memoization stores `F(n-1)` and `F(n-2)` so they are each computed once ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=Memoization%20is%20an%20optimization%20process,memory%20for%20storing%20intermediate%20results)) ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)).

- **Tabulation (Bottom-Up):** Define the states and the recurrence, then iteratively compute solutions for all subproblems up to the one you need. It’s bottom-up because you start from base cases and iteratively reach the final solution. This typically uses an array or table to store results for all states. For example, filling a DP table for Fibonacci from `F(0)` and `F(1)` up to `F(n)`. Bottom-up ensures all needed subproblems are solved in advance. It can be more space-efficient (no recursion stack) and there’s no function call overhead ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=Speed%20Fast%2C%20as%20we%20do,subproblems%20that%20are%20definitely%20required)). However, you might compute some states that a top-down approach would never visit if the problem doesn’t require them ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=Speed%20Fast%2C%20as%20we%20do,subproblems%20that%20are%20definitely%20required)).

In summary, DP is applicable when a problem can be recursively defined in terms of overlapping subproblems with optimal solutions. By carefully defining states and using memoization or tabulation to reuse solutions, we can reduce exponential brute-force solutions to polynomial time ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)). In the next sections, we will explore common DP problem **patterns**, each with typical scenarios, solution tricks, and example problems.

# 2. DP Patterns

There are many classic DP patterns that frequently appear in coding interviews (including Google’s). Recognizing the pattern behind a problem can give insight into the state design and recurrence. Below we cover several major DP patterns. For each pattern, we provide a summary, key techniques, a JavaScript example, and a list of 20 LeetCode example problems (from easy to hard) to practice.

## Fibonacci-Style DP

**Summary:** *Fibonacci-style DP* refers to one-dimensional DP problems where the state progresses linearly (often along a single index or size), and the solution for state `i` depends on a fixed number of previous states (such as `i-1`, `i-2`, etc). These include scenarios like computing Fibonacci numbers, climbing stairs, or the House Robber problem. The name comes from the Fibonacci sequence, which is a simple DP where `F(n) = F(n-1) + F(n-2)`.

Such problems often model something like “how many ways” or “what is the best we can do up to this point,” where the recurrence relation naturally connects to one or two prior states. For example, *Climbing Stairs* (LC 70) asks for the number of ways to reach the top of *n* steps if you can take 1 or 2 steps at a time – the answer for `n` steps is the sum of ways to reach `n-1` and `n-2` steps (Fibonacci sequence) because the last move could have been a 1-step or 2-step climb. Similarly, the *House Robber* problem (LC 198) yields a recurrence `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` – at each house, you either rob it and add to the best up to two houses before, or skip it to take the best up to the previous house ([House Robber LeetCode Solution - To The Innovation](https://totheinnovation.com/house-robber-leetcode-solution/#:~:text=House%20Robber%20LeetCode%20Solution%20,This%20pattern)). This “take one or skip one” structure is Fibonacci-like. 

**Common Techniques and Tricks:** For Fibonacci-style DP, an important trick is **space optimization** – since the state only depends on a few previous values, you don’t need to store the entire DP table. You can keep two or three variables to represent the last states. Many of these problems have straightforward recurrences and base cases; the challenge is often recognizing the pattern. Key steps:
- Identify the recurrence relation. Usually `dp[i] = dp[i-1] + dp[i-2]` (or a variant like addition with conditions, or a `max` of two previous states).
- Set base cases for the first one or two values (e.g. `dp[0]` and `dp[1]`).
- Use a simple loop or recursion with memoization to build up to the solution.
- Optimize space by only storing needed previous states (turning O(n) space into O(1) in many cases) ([How would you solve the knapsack problem using dynamic ... - Taro](https://www.jointaro.com/interview-insights/amazon/how-would-you-solve-the-knapsack-problem-using-dynamic-programming-including-the-time-and-space-complexity-analysis-and-optimizations/#:~:text=Taro%20www,on%20the%20previous%20row)).

This pattern appears in many counting problems (staircase, tiling, decoding messages) and simple optimal choice problems (robbery, game moves) where the transition only looks back a constant number of steps.

**JavaScript Code Example – House Robber (Fibonacci-style DP):** In the House Robber problem, you cannot rob two adjacent houses. The maximal loot up to house *i* depends on either skipping house *i* or robbing house *i* and adding the loot from *i-2*. We can implement this DP in a bottom-up manner:

```js
function rob(nums) {
  const n = nums.length;
  if (n === 0) return 0;
  // dp[i] will hold the max profit for subarray [0..i-1]
  let dp = new Array(n+1);
  dp[0] = 0;            // no houses -> 0 profit
  dp[1] = nums[0];      // only first house -> rob it
  for (let i = 2; i <= n; i++) {
    // either skip current house (take dp[i-1]) 
    // or rob it (nums[i-1] + dp[i-2])
    dp[i] = Math.max(dp[i-1], nums[i-1] + dp[i-2]);
  }
  return dp[n];
}
```

*Comments:* We use `dp[i]` for the maximum loot considering up to the i-th house (1-indexed for convenience, so `dp[1]` corresponds to first house). The recurrence is `dp[i] = max(dp[i-1], dp[i-2] + value[i])`. Base cases: `dp[0]=0` (no houses) and `dp[1] = value of house0`. This is a classic Fibonacci-style relation (it mirrors `dp[i] = max(dp[i-1], dp[i-2] + ...)`). We used an array for clarity, but we could optimize to O(1) space by only keeping the last two values because `dp[i]` only depends on `dp[i-1]` and `dp[i-2]`. For example:

```js
function robOptimized(nums) {
  let prev2 = 0, prev1 = 0;         // prev2 = dp[i-2], prev1 = dp[i-1]
  for (let num of nums) {
    let cur = Math.max(prev1, prev2 + num);
    prev2 = prev1;
    prev1 = cur;
  }
  return prev1;
}
```

Here `prev1` and `prev2` correspond to `dp[i-1]` and `dp[i-2]` during iteration, achieving O(1) space.

**Problem List (LeetCode examples from easy to hard):**
1. **Climbing Stairs** – *LeetCode 70 (Easy)* – Basic Fibonacci DP ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)).
2. **Fibonacci Number** – *LeetCode 509 (Easy)* – Direct computation of F(n) with DP.
3. **Min Cost Climbing Stairs** – *LeetCode 746 (Easy)* – Similar to Climbing Stairs but accumulate cost.
4. **N-th Tribonacci Number** – *LeetCode 1137 (Easy)* – Like Fibonacci but depends on the last 3 terms.
5. **Maximum Subarray** – *LeetCode 53 (Easy)* – Uses Kadane’s DP: `dp[i] = max(nums[i], dp[i-1]+nums[i])`.
6. **House Robber** – *LeetCode 198 (Medium)* – Rob linear houses (shown above).
7. **House Robber II** – *LeetCode 213 (Medium)* – Rob houses in a circle (modify endpoints handling).
8. **Delete and Earn** – *LeetCode 740 (Medium)* – Convert to House Robber pattern by re-grouping values.
9. **Paint House** – *LeetCode 256 (Medium)* – DP on a line of houses with 3-color minimum painting cost.
10. **Decode Ways** – *LeetCode 91 (Medium)* – Count ways to decode a digit string (`dp[i] = (s[i] valid ? dp[i-1]:0) + (s[i-1..i] valid ? dp[i-2]:0)`).
11. **Jump Game** – *LeetCode 55 (Medium)* – Can you reach the end of array with jumps (Greedy preferred, but DP possible by marking reachable indices).
12. **Jump Game II** – *LeetCode 45 (Medium)* – Minimum jumps to reach end (can use BFS/DP; greedy optimal).
13. **Frog Jump** – *LeetCode 403 (Hard)* – Can a frog reach the final stone? DP by set of achievable jumps for each stone.
14. **Arithmetic Slices** – *LeetCode 413 (Medium)* – Count subarrays forming arithmetic progression (dp for consecutive slices).
15. **Ugly Number II** – *LeetCode 264 (Medium)* – Generate sequence with DP and pointers (each term is min(prev2*2, prev3*3, prev5*5)).
16. **Catalan Numbers (Unique BSTs)** – *LeetCode 96 (Medium)* – Number of unique BSTs with `n` nodes (DP or formula).
17. **Minimum Cost Tickets** – *LeetCode 983 (Medium)* – DP over days with choices of 1-day, 7-day, 30-day passes (linear DP with look-back).
18. **Decode Ways II** – *LeetCode 639 (Hard)* – Like Decode Ways but `*` wildcard digits, more complex DP.
19. **Paint Fence** – *LeetCode 276 (Medium)* – Number of ways to paint fence with no more than 2 adjacent of same color (linear DP).
20. **Cherry Pickup II** – *LeetCode 1463 (Hard)* – 2 robots collecting grid rewards (2D grid but can reduce to DP per row with states, complex but follows from smaller overlapping transitions).

## Knapsack Pattern (0/1 and Unbounded)

**Summary:** The *Knapsack pattern* covers DP problems where you make a choice to include or exclude an item (or take it multiple times) under certain constraints. This pattern is named after the classic **0/1 Knapsack problem**: given a set of items with values and weights, pick a subset within a weight capacity to maximize total value. In 0/1 knapsack each item can be taken at most once (include or not include – the “0/1” choice) ([Algorithm Analysis: Week 9](https://cs.franklin.edu/~shaffstj/cs319/week9.htm#:~:text=Algorithm%20Analysis%3A%20Week%209%20In,the%20profit%20of%20the)) ([DSA The 0/1 Knapsack Problem](https://www.w3schools.com/dsa/dsa_ref_knapsack.php#:~:text=,of%20an%20item%20for%20example)). A related variant is **unbounded knapsack**, where you can take items repeatedly (infinite supply), which is the basis for coin change and similar problems.

Problems in this category typically involve a trade-off decision: *take something and incur a cost/gain, or skip it*. They often ask for an optimal value (max or min) achievable under constraints, or a boolean answer whether a certain sum/target is achievable. Besides the literal knapsack problem, this pattern applies to many scenarios:
- **Subset selection for target sum or partition:** e.g. choose numbers that sum to a target (see next pattern).
- **Minimizing or maximizing a value with choices:** e.g. coin change (min coins to make amount, or number of ways to make amount), making change for a value, or selecting projects within a budget for maximum profit.
- **Resource allocation problems:** limited resources (weight, time, capacity) and items or tasks with benefits.

**Common Techniques and Tricks:** Knapsack DP generally uses a DP table where one dimension represents considering first *i* items and another represents the capacity or target. A classic 0/1 knapsack recurrence is: 

`dp[i][w] = max(dp[i-1][w],  dp[i-1][w - weight_i] + value_i)`,

meaning for item *i*, you either skip it (value stays as using first *i-1* items for weight *w*) or take it (value = value_i + solution for remaining capacity *w - weight_i*) ([Algorithm Analysis: Week 9](https://cs.franklin.edu/~shaffstj/cs319/week9.htm#:~:text=Algorithm%20Analysis%3A%20Week%209%20In,the%20profit%20of%20the)). Base cases are zero items or zero capacity. For *unbounded* knapsack (e.g. coin change combinations), you often allow using the same item again by not moving to the next item in the recurrence (or loop items outside weight loop).

**Tricks:** 
- For 0/1 knapsack, iterate weight capacity in descending order when using a 1D array to avoid using one item multiple times (process from high weight to low so that when you use `dp[w-weight_i]` it refers to previous iteration’s value) ([How would you solve the knapsack problem using dynamic ... - Taro](https://www.jointaro.com/interview-insights/amazon/how-would-you-solve-the-knapsack-problem-using-dynamic-programming-including-the-time-and-space-complexity-analysis-and-optimizations/#:~:text=Taro%20www,on%20the%20previous%20row)).
- For unbounded knapsack (infinite supply), iterate capacity in ascending order (so you can reuse the same item in the same iteration).
- Sometimes use bitsets or boolean DP for subset-sum style problems to optimize space.
- Watch out for scenarios with multiple constraints (e.g. “Ones and Zeroes” problem has two resources). They often generalize knapsack to higher dimensions (2D DP array for two constraints).
- If only a yes/no answer is needed (feasible or not), a boolean DP can be used. If counting ways, use addition in recurrence (and possibly large modulo arithmetic).
- Many problems reduce to knapsack variants with transformations. E.g., *Partition Equal Subset Sum* is a knapsack where each number is an item and target weight is half the total sum.

**JavaScript Code Example – 0/1 Knapsack:** The code below implements the classic knapsack using a 2D DP table. `dp[i][w]` = maximum value using first `i` items within weight `w`.

```js
function knapSack(weights, values, W) {
  const n = weights.length;
  // dp table of size (n+1) x (W+1), initialized with 0
  let dp = Array.from({length: n+1}, () => Array(W+1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let w = 0; w <= W; w++) {
      if (weights[i-1] <= w) {
        // Option1: take item i-1, Option2: skip it
        dp[i][w] = Math.max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]]);
      } else {
        // Can't take item i-1 (exceeds capacity)
        dp[i][w] = dp[i-1][w];
      }
    }
  }
  return dp[n][W];  // max value for n items and capacity W
}
```

*Comments:* We use a 2D table where rows represent considering up to that item, and columns represent capacity from 0 to W. If the current item (index `i-1`) can fit in the remaining capacity `w`, we decide to take it or not. If it doesn’t fit, we carry forward the value without it. This runs in O(n*W) time. For space optimization, we could use a 1D array of size `W+1` and loop weights backward for 0/1 case. For example:

```js
function knapSack1D(weights, values, W) {
  const n = weights.length;
  let dp = Array(W+1).fill(0);
  for (let i = 0; i < n; i++) {
    for (let w = W; w >= weights[i]; w--) {
      dp[w] = Math.max(dp[w], values[i] + dp[w - weights[i]]);
    }
  }
  return dp[W];
}
```

This uses only one row which we update in reverse order to simulate the 2D DP.

**JavaScript Code Example – Coin Change (Unbounded Knapsack):** In the Coin Change problem, we want the fewest coins to make a given amount. This is an *unbounded* knapsack (we can use coin types unlimited times) aiming to minimize cost. We can use a 1D DP where `dp[x]` = min coins to make amount `x`. We iterate coins in outer loop (to allow unlimited use of each coin) and amount in inner loop:

```js
function coinChange(coins, amount) {
  let dp = Array(amount+1).fill(Infinity);
  dp[0] = 0;
  for (let coin of coins) {
    for (let x = coin; x <= amount; x++) {
      dp[x] = Math.min(dp[x], dp[x-coin] + 1);
    }
  }
  return dp[amount] === Infinity ? -1 : dp[amount];
}
```

*Comments:* We initialize `dp[0]=0` (0 coins to make 0) and Infinity for others (meaning not reachable yet). For each coin, we update the dp table for all amounts ≥ coin value. The relation is `dp[x] = min(dp[x], 1 + dp[x-coin])`. This is unbounded: we can use the coin again since we do `dp[x-coin]` which may have been updated in the same coin’s iteration (hence iterating `x` from `coin` to `amount`). The result is `dp[amount]` or -1 if it stayed Infinity (amount not reachable).

**Problem List (LeetCode examples from easy to hard):**
1. **Coin Change** – *LeetCode 322 (Medium)* – Minimum number of coins to make a given amount (unbounded, minimize) ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=Speed%20Fast%2C%20as%20we%20do,subproblems%20that%20are%20definitely%20required)).
2. **Coin Change 2** – *LeetCode 518 (Medium)* – Count the number of ways to make amount (unbounded, count combinations).
3. **Perfect Squares** – *LeetCode 279 (Medium)* – Fewest perfect square numbers summing to *n* (unbounded, like coin change with squares).
4. **Minimum Cost For Tickets** – *LeetCode 983 (Medium)* – Min cost to cover travel days with 1-day, 7-day, 30-day passes (can be done with DP like coin change or sliding window).
5. **Ones and Zeroes** – *LeetCode 474 (Medium)* – 0/1 knapsack with two weights (limit of 0s and 1s you can use, maximize number of strings chosen).
6. **Partition Equal Subset Sum** – *LeetCode 416 (Medium)* – Determine if you can pick a subset that sums to half of total (0/1 knap as boolean) ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=,be%20solved%20using%20Dynamic%20Programming)).
7. **Last Stone Weight II** – *LeetCode 1049 (Medium)* – Partition set into two to minimize difference (equivalent to subset sum close to half).
8. **Target Sum** – *LeetCode 494 (Medium)* – Count ways to assign +/- signs to make a target (can be transformed to subset count problem).
9. **Combination Sum** – *LeetCode 39 (Medium)* – Find all combinations of candidates summing to target (backtracking, but akin to unbounded knapsack enumeration).
10. **Combination Sum IV** – *LeetCode 377 (Medium)* – Count permutations of numbers that sum to target (unbounded, order matters – a slight permutation twist on coin change).
11. **Profitable Schemes** – *LeetCode 879 (Hard)* – 0/1 knapsack with group members as one weight and profit as value, with an additional profit target constraint (DP in two dimensions: people and profit).
12. **Form Largest Integer With Digits That Add up to Target** – *LeetCode 1449 (Hard)* – Choose digits (like items) to reach a target sum (target = sum of digits) while maximizing the formed number’s value (lexicographically). Uses knapSack-style DP with custom comparison for maximizing numeric string.
13. **Matchsticks to Square** – *LeetCode 473 (Medium)* – Partition sticks into 4 equal subsets (can be solved via DFS + memo or bitmask DP; it’s essentially subset-sum repeated).
14. **Partition to K Equal Sum Subsets** – *LeetCode 698 (Hard)* – Generalized partition into *k* equal subsets (often solved with DFS + memoization, subset DP or bitmask DP).
15. **Minimum Number of Refueling Stops** – *LeetCode 871 (Hard)* – Can be seen as a knapsack where stops are like items that extend reach; solved with DP where `dp[t]` = farthest distance with t stops (optimize fuel usage under stop limit).
16. **Booleans to String (Word Break)** – *LeetCode 139 (Medium)* – Determine if a string can be segmented into dictionary words (can treat each dictionary word as an item that “fills” a portion of the target string length).
17. **Word Break II** – *LeetCode 140 (Hard)* – Return all possible segmentations (uses DFS+DP to store results for substrings).
18. **Minimum Partition Sum Difference** – *Not on LeetCode (Medium)* – Classic partition problem (covered by LC1049 Last Stone Weight II).
19. **Subset Sum Problem** – *Not on LeetCode (Easy)* – Determine if any subset sums to a given S (basis of 416).
20. **0/1 Knapsack Problem** – *Not on LeetCode (Medium)* – The classic formulation for reference (maximize value under weight constraint).

## Subset Sum / Partition DP

**Summary:** This pattern is a specialized case of knapsack where we focus on using a subset of numbers to meet a certain sum condition, often without explicit “values” for optimization beyond achieving the sum. Typical problems ask if a subset exists that meets a criterion (true/false), or to count the number of such subsets. The most common examples are *Subset Sum* (determine if any subset of a given set sums to a target) and *Partition problems* like splitting an array into two equal-sum subsets or subsets with minimum difference. Unlike the general knapsack, these problems usually treat each number as either taken or not (0/1 choice) without distinct “value vs weight” distinction – the numbers themselves represent the “weight” or sum. 

**Common Techniques and Tricks:** A common DP formulation for subset-sum is a boolean DP table `dp[i][s]` meaning “using first i numbers, can we reach sum s?”. The recurrence is: either we don’t use the i-th number (carry over the possibility from `dp[i-1][s]`), or we use it (if `dp[i-1][s-num_i]` was true, then `dp[i][s]` becomes true) ([Overlapping Subproblems Property in Dynamic Programming | DP-1 - GeeksforGeeks](https://www.geeksforgeeks.org/overlapping-subproblems-property-in-dynamic-programming-dp-1/#:~:text=,are%20solved%20again%20and%20again)). Base case: `dp[0][0] = true` (sum 0 is reachable with no elements). This DP runs in O(n * targetSum). For partition into equal subsets, targetSum is total/2.

For efficiency, a 1D boolean array can be used (of size target+1) where you iterate through numbers and update achievable sums. **Trick:** iterate the sum backwards (from target down to num) when using 1D, so each number is only counted once. This is similar to knapsack optimization. The 1D `dp[s]` will represent whether sum `s` can be formed by some subset. 

For counting subsets (e.g. count subsets that sum to target), use an integer DP array and sum up counts. For example, `count[s] += count[s - num]` for each number (iterating s downward for 0/1 case).

Some problems, like *Target Sum* (which asks to assign +/- to make a target), can be transformed into a subset-sum count problem by algebra (partition into two groups with a certain difference). Others like *Partition to K Equal Subsets* use DFS with memo (which is effectively DP over subsets via bitmask). In general, if the set size or target sum is not too large, subset DP or bitmask DP is a powerful tool for these problems.

**Common pitfalls:** Watch out for off-by-one errors in indexing, and remember to initialize `dp[0] = true` (sum 0 achievable with no elements). For counting, be careful with order of iteration to avoid double counting. If numbers can be 0 or negative (usually they aren’t in subset-sum classics), the state space or logic may need adjustments.

**JavaScript Code Example – Partition Equal Subset Sum:** We determine if the array can be partitioned into two subsets of equal sum (which requires total sum to be even, and then a subset with sum = total/2 exists). We use 1D DP for subset sum:

```js
function canPartition(nums) {
  const total = nums.reduce((a,b) => a + b, 0);
  if (total % 2 !== 0) return false;
  const target = total / 2;
  let dp = new Array(target + 1).fill(false);
  dp[0] = true;
  for (let num of nums) {
    for (let s = target; s >= num; s--) {
      if (dp[s - num]) {
        dp[s] = true;
      }
    }
  }
  return dp[target];
}
```

*Comments:* Here `dp[s]` indicates whether a subset with sum `s` is achievable. We initialize `dp[0]=true`. For each number, we update the dp array from right to left (from `target` down to `num`) setting `dp[s] = true` if `dp[s-num]` was true (meaning we can form `s-num` before, so now we can form `s` by adding this number). We iterate backward to ensure each number is only used once. In the end, we check `dp[target]`. This runs in O(n * target) time. If `target` is at most 20000 or so (for example,  
 constraint sums), this is efficient. This same code can answer a generic subset sum question for target.

**JavaScript Code Example – Target Sum (Count of Subsets):** In Target Sum (LC 494), we want to count the number of ways to assign +/– to achieve a given total. This can be transformed to: count subsets of `nums` that have sum `(total + target)/2` (where `total` is sum of all nums, provided `(total+target)` is even) ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=,be%20solved%20using%20Dynamic%20Programming)). Below we use a DP count approach for subset sum counts:

```js
function findTargetSumWays(nums, S) {
  const total = nums.reduce((a,b)=>a+b, 0);
  // if S is not achievable or parity issue, return 0
  if (S > total || (total + S) % 2 === 1) return 0;
  const target = (total + S) / 2;
  let dp = new Array(target + 1).fill(0);
  dp[0] = 1;
  for (let num of nums) {
    for (let s = target; s >= num; s--) {
      dp[s] += dp[s - num];
    }
  }
  return dp[target];
}
```

*Comments:* This is similar to subset sum, but `dp[s]` holds the **number of ways** to form sum `s`. We initialize `dp[0]=1` (one way to have sum 0: choose nothing). We accumulate counts: for each number, iterate `s` downwards and do `dp[s] += dp[s-num]`. After processing all numbers, `dp[target]` gives the count of subsets that sum to `target`, which corresponds to one half of the partitions that achieve difference S. This solves Target Sum efficiently by leveraging subset-sum counting.

**Problem List (LeetCode examples from easy to hard):**
1. **Subset Sum (Basic)** – *No direct LeetCode problem, but fundamental* – Decide if any subset sums to a target (basis of many others).
2. **Partition Equal Subset Sum** – *LeetCode 416 (Medium)* – Decide if array can split into two equal-sum parts (subset sum to total/2).
3. **Last Stone Weight II** – *LeetCode 1049 (Medium)* – Partition multiset into two with minimal difference (subset sum close to half).
4. **Target Sum** – *LeetCode 494 (Medium)* – Count subsets that can achieve a given difference (DP counting approach as above).
5. **Partition to K Equal Sum Subsets** – *LeetCode 698 (Hard)* – Can array be partitioned into k subsets of equal sum (typically k=4 or general k, solved via DFS + memo or DP on bitmasks of used elements).
6. **Matchsticks to Square** – *LeetCode 473 (Medium)* – A specific case of k-partition with k=4 (partition into 4 equal sums, often solved similarly to #5).
7. **Split Array with Same Average** – *LeetCode 805 (Hard)* – Determine if array can be split into two subsets of equal average (requires subset of certain sum and size – can be solved by DP on possible sums and counts via bitset or DP).
8. **Bitmask Subset Sum** – *Conceptual* – (Using bitsets for subset sum is another optimization trick: e.g. using a bitset where bit shifts simulate addition).
9. **Count of Subset Sum** – *Variant* – Count subsets that sum to a given value (e.g. Target Sum can be transformed to this, or some interview questions directly ask for count).
10. **Minimum Subset Sum Difference** – *Equivalent to Last Stone Weight II* – Find two subsets with min difference (solved by computing reachable sums and checking nearest to total/2).
11. **Partition Array Into Two Arrays to Minimize Sum Difference** – *LeetCode 2035 (Medium)* – Similar partition problem for minimum difference (with potentially larger constraints, often using meet-in-middle or optimized DP).
12. **Can I Win** – *LeetCode 464 (Medium)* – Though not a sum problem per se, it can be modeled as a subset reachability: players choose numbers 1–n, trying to reach a total >= target. Often solved with DFS + memo (bitmask representing chosen numbers), effectively exploring subsets of moves.
13. **Stone Game (pile partition)** – *LeetCode 877 (Medium)* – Although usually solved with game DP, the case with equal piles could be seen as partition (but generally this is more game theory).
14. **Subset Sum (Bitset optimization)** – *LeetCode 1043 (not exactly subset sum, skip)*
15. **Balanced Partition (GFG)** – *Similar to 416* – classic DP subset difference.
16. **Circular Partition** – *No specific problem, concept: e.g. splitting circular arrangement, could reduce to linear by doubling or so.*
17. **Concatenated Words** – *LeetCode 472 (Hard)* – Check if a word can be formed by concatenating other words from a list. (This is a *subset* of words forming another word – solved by DP over the word string length, marking valid breaks similar to word break.)
18. **Word Break** – *LeetCode 139 (Medium)* – Determine if a string can be segmented into dictionary words (DP on substring validity, analogous to subset sum where “target” is the full string length and “pieces” are word lengths that fit).
19. **Word Break II** – *LeetCode 140 (Hard)* – Find all ways to segment a string (backtracking with memoization – expands on the DP idea of Word Break).
20. **Palindrome Partitioning** – *LeetCode 131 (Medium)* – Partition a string into palindromic substrings (asks for all partitions – solved with DFS, but can use DP to precompute palindrome substrings). *(This is more of a backtracking problem, but involves a partition concept.)*

## Counting Ways / Combinatorial DP

**Summary:** This pattern is about counting the number of ways to do something, often under some constraints. Many DP problems ask not for an optimal value, but *how many sequences/structures satisfy the conditions*. These are combinatorial DP problems. Examples include: count of ways to climb stairs (which is Fibonacci DP), number of ways to coin change, count of paths in a grid, number of ways to decode a message, etc. What distinguishes these from other patterns is that we are summing up counts rather than taking max/min, and the DP often accumulates *all possible ways* that subproblems can occur.

Typical scenarios:
- **Combinatorial paths or sequences:** e.g. number of ways to reach a target (climbing stairs, reaching a sum, forming a string).
- **Partition counts:** e.g. how many ways to partition a set or number (like coin change combinations).
- **Dynamic counting in grids:** unique paths in a grid, ways to place tiles, etc.
- **Permutations and combinations with DP:** e.g. counting permutations meeting certain conditions (like the *Decode Ways* problem or *Combination Sum IV* which counts ordered combinations).

**Common Techniques and Tricks:** 
- These problems often have recurrences that add up contributions from previous states rather than taking min/max. For instance, *Unique Paths in a Grid* has `dp[i][j] = dp[i-1][j] + dp[i][j-1]` (paths from top + paths from left) ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)), and *Coin Change 2* has a similar additive formula.
- It’s important to set initial counts properly. Typically `dp[0] = 1` (one way to achieve “zero” – by choosing nothing).
- Order matters: If the problem considers two sequences different even if they contain the same elements in different order (permutations), the DP must account for order (usually by iterating states in the opposite way or using a different approach). For example, Combination Sum IV (counting sequences of numbers that sum to target) treats `[1,2]` and `[2,1]` as different, so the DP there often iterates over target sum first and then numbers.
- Avoid double counting: When counting combinations (order doesn’t matter), ensure each combination is counted once (often by iterating items first then sum, like unbounded knapsack combination approach). When counting permutations (order matters), a different ordering of loops or including sequence position in state might be needed.
- Use modular arithmetic if the counts can be very large (some problems ask for answer mod 1e9+7).
- Many of these problems are similar to *Fibonacci-style* in recurrence (since Fibonacci itself is a count of ways problem), except with branching factors or more dimensions.

**JavaScript Code Example – Unique Paths in Grid:** We count the number of distinct paths from the top-left to bottom-right of an `m x n` grid, moving only down or right. This is a classic combinatorial DP:

```js
function uniquePaths(m, n) {
  // dp[i][j] = number of ways to reach cell (i,j)
  let dp = Array.from({length: m}, () => Array(n).fill(0));
  // first row and first column can only be reached in one way
  for (let i = 0; i < m; i++) dp[i][0] = 1;
  for (let j = 0; j < n; j++) dp[0][j] = 1;
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      dp[i][j] = dp[i-1][j] + dp[i][j-1];
    }
  }
  return dp[m-1][n-1];
}
```

*Comments:* We initialize the first row and column to 1 because there is exactly one way to reach any cell in the top row (all moves right) or left column (all moves down) from the start. Then each other cell’s ways = ways from above + ways from left ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)). This yields the total number of paths. (This formula corresponds to binomial coefficients as well, but the DP provides an intuitive solution.)

**JavaScript Code Example – Decode Ways:** This is a one-dimensional DP (similar to Fibonacci) but with conditional additions. Each digit can be taken alone or (if valid) paired with the previous digit to form a letter:

```js
function numDecodings(s) {
  const n = s.length;
  if (n === 0) return 0;
  let dp = new Array(n+1).fill(0);
  dp[0] = 1;  // empty string has 1 way to decode (base case)
  dp[1] = s[0] !== '0' ? 1 : 0;  // first char valid decode or not
  for (let i = 2; i <= n; i++) {
    let oneDigit = parseInt(s[i-1]);      // last one digit
    let twoDigits = parseInt(s[i-2] + s[i-1]);  // last two digits
    if (oneDigit !== 0) {
      dp[i] += dp[i-1];
    }
    if (twoDigits >= 10 && twoDigits <= 26) {
      dp[i] += dp[i-2];
    }
  }
  return dp[n];
}
```

*Comments:* `dp[i]` represents number of ways to decode up to the i-th character. If the last single digit is valid (non-zero), we add ways from `dp[i-1]`. If the last two-digit number forms a valid letter (10 to 26), we add ways from `dp[i-2]`. This is essentially summing contributions of each possible decision (use one digit or two digits as a chunk). It’s similar to Fibonacci but with additional conditional branches.

**Problem List (LeetCode examples from easy to hard):**
1. **Climbing Stairs** – *LeetCode 70 (Easy)* – Count ways to climb n stairs (each step 1 or 2) – results in Fibonacci numbers.
2. **Unique Paths** – *LeetCode 62 (Medium)* – Count paths in an m×n grid (simple combinatorial DP).
3. **Unique Paths II** – *LeetCode 63 (Medium)* – Count paths in a grid with obstacles (same as above but treat obstacle cells as 0 ways).
4. **Minimum Path Sum** – *LeetCode 64 (Medium)* – (Not counting ways but summing costs – still grid DP, but optimization rather than counting).
5. **Decode Ways** – *LeetCode 91 (Medium)* – Count decodings of a numeric string (shown above).
6. **Coin Change 2** – *LeetCode 518 (Medium)* – Count combinations of coins to make amount (unbounded knapsack counting).
7. **Combination Sum IV** – *LeetCode 377 (Medium)* – Count permutations of numbers that sum to target (order matters, requires careful DP).
8. **Possible Bipartitions** – *LeetCode 886 (Medium)* – (Graph problem; skip – not DP).
9. **BoB Tiling Problem** – e.g. *Domino and Tromino Tiling* – *LeetCode 790 (Medium)* – Count ways to tile a 2xN board with 2x1 dominoes and L trominoes (linear DP).
10. **Binary Tree Traversals Count** – e.g. *Unique BSTs* – *LeetCode 96 (Medium)* – Count structurally unique BSTs (Catalan numbers, DP formula).
11. **Interleaving String (Count)** – Variation if asked to count interleavings (though LC 97 is just boolean check).
12. **Count Vowel Strings** – *LeetCode 1641 (Medium)* – Count sorted vowel strings of length n (combinatorial DP).
13. **Dice Roll Sum** – *LeetCode 1155 (Medium)* – Count ways to roll dice to reach a target sum (unbounded-like DP with limited 6 faces each roll).
14. **Painting Fence** – (If counting arrangements with no more than 2 adjacent same colors, can derive formula with DP).
15. **Bell Numbers (Count of set partitions)** – not on LC, but combinatorial DP.
16. **Probability DP** – e.g. *Knight Probability in Chessboard* – *LeetCode 688 (Medium)* – (Count ways or probability of reaching positions in K moves, uses DP to accumulate probabilities).
17. **Catalan DP problems** – e.g. *Valid Parentheses combinations* (LC 22 just asks to generate, but count would be Catalan number).
18. **Permutations with restrictions** – e.g. *Beautiful Arrangement* – *LeetCode 526 (Medium)* – Count permutations of 1..N satisfying certain divisibility (often solved with DFS + DP).
19. **Harder Grid Paths** – e.g. *Avoiding Obstacles and Counting Paths* – (No specific LC, but variations of Unique Paths with restrictions).
20. **Advanced counting** – e.g. *Palindrome Partitioning II (count partitions)* or *Counting Subsequences* – *LeetCode 940 (Hard)* – Count distinct subsequences in a string (DP with combinatorial counting and modulo).

## Longest Increasing Subsequence (LIS) and Variants

**Summary:** This pattern involves finding a longest subsequence (not necessarily contiguous) or chain that satisfies a certain property, typically monotonic increasing order. The classic problem is **Longest Increasing Subsequence**: given an array, find the length of the longest strictly increasing subsequence. This can be solved with DP in O(n^2) time (or O(n log n) with a different method). Variants include finding the sequence itself, finding the longest *non-decreasing* subsequence, counting the number of LIS, or related problems like *Longest Chain of Pairs*, *Longest Increasing Path in a matrix* (which is a 2D variant), *Longest Bitonic Subsequence* (increase then decrease), etc.

The common thread is a DP where each state depends on earlier states that represent a smaller element. Typically, `dp[i]` represents the length of the longest increasing subsequence *ending at index i*. Then the recurrence is: for all `j < i` if `nums[j] < nums[i]`, consider subsequences ending at `j` and extend them with `nums[i]`. So `dp[i] = 1 + max(dp[j])` for all `j` with `nums[j] < nums[i]` ([Longest increasing subsequence](https://cp-algorithms.com/sequences/longest_increasing_subsequence.html#:~:text=Longest%20increasing%20subsequence%20First%20we,to%20restore%20the%20subsequence%20itself)). If none, then `dp[i] = 1` at least itself. The LIS length is then `max(dp[i])` over all i.

**Common Techniques and Tricks:** The straightforward DP is O(n^2) as described. There is a well-known optimization using a sorted list or binary search (Patience Sorting technique) to get O(n log n) for LIS length, but that is a different algorithm (still conceptually DP but not table-filling). If the problem just asks for length, the binary search method is preferred for efficiency when n is large (like > 2500). If it asks for the sequence or count of sequences, the O(n^2) DP is often needed (with backtracking for sequence, or parallel DP for counts).

Variants:
- **LIS length**: DP or patience sorting.
- **LIS actual sequence**: maintain predecessor pointers in DP to reconstruct the path.
- **Number of LIS**: maintain another array `count[i]` where `count[i]` is number of LIS ending at i. When extending from j to i (nums[j] < nums[i] and dp[j] + 1 = dp[i]), accumulate counts.
- **Longest Bitonic Subsequence**: find LIS ending at each index and LDS (longest decreasing subseq) starting at each index, then combine for each peak.
- **Longest Chain of Pairs** (like pairs (a,b) sorted by first, find longest chain by second): similar to LIS after sorting.
- **Longest Increasing Path in Matrix**: 2D DFS with memo, essentially LIS in a directed acyclic graph defined by matrix adjacencies.
- **Maximum Envelopes (Russian Doll Envelopes)** – sort envelopes by one dimension and LIS by the other dimension.

**Tricks:** 
- For LIS in array: If n is up to ~1000 or 2000, O(n^2) DP is fine. For n larger (like 10^4, 10^5), use the patience sorting (maintain an array where `tails[len]` = smallest possible tail value of an increasing subsequence of length `len`; binary search to place each element).
- Watch out for equality: usually “increasing” means strictly, but sometimes non-decreasing is allowed (adjust condition accordingly).
- When reconstructing LIS, store predecessor indices. When counting LIS, be careful to reset counts when finding a longer subsequence, or add counts when finding another way to get same length.

**JavaScript Code Example – Longest Increasing Subsequence (O(n²) DP):**

```js
function lengthOfLIS(nums) {
  const n = nums.length;
  if (n === 0) return 0;
  let dp = new Array(n).fill(1);
  let maxLen = 1;
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) {
      if (nums[j] < nums[i]) {
        dp[i] = Math.max(dp[i], dp[j] + 1);
      }
    }
    maxLen = Math.max(maxLen, dp[i]);
  }
  return maxLen;
}
```

*Comments:* We initialize all `dp[i] = 1` (at least the element itself). We then loop `i` from 1 to n-1, and for each look at all previous `j < i`. If `nums[j] < nums[i]`, we can append `nums[i]` after the LIS ending at `j`. We take the best among those. We keep track of a global `maxLen`. This is a classic LIS DP ([Longest increasing subsequence](https://cp-algorithms.com/sequences/longest_increasing_subsequence.html#:~:text=Longest%20increasing%20subsequence%20First%20we,to%20restore%20the%20subsequence%20itself)). It is O(n²). 

We could optimize with a separate array `tails` for patience sorting:
```js
function lengthOfLIS_fast(nums) {
  let tails = [];
  for (let num of nums) {
    // find insertion point in tails (binary search)
    let l = 0, r = tails.length;
    while (l < r) {
      let mid = (l + r) >> 1;
      if (tails[mid] < num) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    tails[l] = num;
    if (l === tails.length) tails.push(num);
  }
  return tails.length;
}
```
This maintains `tails` such that length of `tails` is the length of LIS found so far ([Longest increasing subsequence](https://cp-algorithms.com/sequences/longest_increasing_subsequence.html#:~:text=Longest%20increasing%20subsequence%20First%20we,to%20restore%20the%20subsequence%20itself)). Each number either extends the LIS (if larger than all tails) or replaces the first tail ≥ it. (Note: The condition here is `< num` for strictly increasing. For non-decreasing LIS, adjust to `<= num`.)

**Problem List (LeetCode examples from easy to hard):**
1. **Longest Increasing Subsequence** – *LeetCode 300 (Medium)* – Length of LIS in an array (classic DP or patience sorting) ([Longest increasing subsequence](https://cp-algorithms.com/sequences/longest_increasing_subsequence.html#:~:text=Longest%20increasing%20subsequence%20First%20we,to%20restore%20the%20subsequence%20itself)).
2. **Longest Increasing Subsequence II** – *LeetCode 1818 (Hard)* – (if exists variant with bigger constraints requiring Fenwick tree or segtree for LIS).
3. **Patience Sorting concept** – *Not a problem but technique for LIS*.
4. **Longest Increasing Path in a Matrix** – *LeetCode 329 (Hard)* – Find length of increasing path in grid (DFS + memo, a form of LIS in 2D).
5. **Longest Bitonic Subsequence** – *LeetCode 376 (Medium, “Wiggle Subsequence”)* – Wiggle subsequence is like a restricted bitonic sequence problem (greedy solution exists for wiggle).
6. **Wiggle Subsequence** – *LeetCode 376 (Medium)* – Length of longest alternating up/down subsequence (can solve with DP for up and down sequences).
7. **Russian Doll Envelopes** – *LeetCode 354 (Hard)* – LIS of envelopes sorted by one dimension (requires sorting and then LIS on the other dimension, tricky because of equal widths – usually solve with patience sorting variant).
8. **Maximum Length of Pair Chain** – *LeetCode 646 (Medium)* – Given pairs, find longest chain (sort by first or second then LIS logic).
9. **Longest Common Increasing Subsequence** – (Not on LeetCode, but a combination of LIS and LCS – solved by DP or patience after intersection).
10. **Length of LIS in a circular sequence** – (Variations where sequence wraps around – can duplicate array and handle overlaps).
11. **Longest Non-Decreasing Subsequence** – (Same as LIS but allow equal, easy tweak).
12. **Number of Longest Increasing Subsequence** – *LeetCode 673 (Medium)* – Count how many LIS are there (DP with count array).
13. **Longest Continuous Increasing Subsequence** – *LeetCode 674 (Easy)* – Consecutive elements increasing (simple linear scan, not really DP needed).
14. **Increasing Triplet Subsequence** – *LeetCode 334 (Medium)* – Just need to detect if LIS of length ≥3 exists (can be done greedily by tracking two values).
15. **Longest Palindromic Subsequence** – *LeetCode 516 (Medium)* – (Different pattern – palindromic, covered in next section).
16. **Longest Arithmetic Subsequence** – *LeetCode 1027 (Medium)* – Length of longest subsequence with constant difference (solved with DP keeping track of differences in a hashmap).
17. **Longest Fibonacci-like Subsequence** – *LeetCode 873 (Medium)* – Length of longest Fib-like subseq in array (DP with pair sums).
18. **Longest String Chain** – *LeetCode 1048 (Medium)* – Longest chain of words where each is formed by adding one letter to previous (sort by length, then DP similar to LIS by checking if a word can follow another).
19. **Longest Increasing Subsequence in a Graph** – (E.g. directed acyclic graph – topologically sort then LIS, or DFS with memo).
20. **Box Stacking / Envelope nesting** – variations where you sort by multiple dimensions and find longest nestable sequence (similar to Russian Doll).

## Palindromic Subsequence / Substring DP

**Summary:** This pattern deals with strings (or sequences) and palindromic properties. Two classic problems are **Longest Palindromic Subsequence (LPS)** and **Longest Palindromic Substring**. They are often solved by DP in O(n²) time. Palindromic-subsequence problems typically use a two-dimensional DP (since checking subsequences from i to j) whereas palindromic-substring problems might use 2D DP or expand-around-center. Another related problem is **Palindromic Partitioning** (cut the string into minimum palindromic pieces) which also uses palindrome checks and DP.

- **Longest Palindromic Subsequence:** find the length of the longest subsequence of a string that is a palindrome. This is solved by DP[i][j] meaning the LPS length in the substring from index i to j. Recurrence: if `s[i] == s[j]`, then they can be endpoints of a palindrome, so `dp[i][j] = 2 + dp[i+1][j-1]`; if they don’t match, `dp[i][j] = max(dp[i+1][j], dp[i][j-1])`. Base case: length 1 substrings have LPS = 1 (a single char) ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=,Output)), and empty string LPS = 0.

- **Longest Palindromic Substring:** find the longest contiguous substring which is a palindrome. A DP approach would be: `dp[i][j]` is true if substring i..j is a palindrome. Then `dp[i][j] = (s[i]==s[j]) AND dp[i+1][j-1]`. Base cases: all single chars `dp[i][i] = true`, and check pairs `dp[i][i+1]` for equality. This DP allows finding the longest span that is true. However, expanding around center (for each center, expand outward while palindrome) is often simpler for substrings.

- **Palindrome Partitioning (min cuts):** You can use a DP where `cut[i]` = min cuts for substring [0..i], and use a palindrome DP or center-expansion to check palindromes quickly.

The key observation in these problems is the **reversal symmetry of palindromes** – it introduces dependencies that make straightforward DP either 2D or somewhat complex.

**Common Techniques and Tricks:**
- Use caching for palindrome checks. If doing palindrome substring DP, often precompute a boolean table `isPal[i][j]` with DP or expand-around-center to be used in other problems (like partitioning).
- Note that subsequence and substring are different: subsequence is not required to be contiguous, so LPS uses a two-index DP expanding outward, whereas substring problems often shrink inward or expand outward.
- LPS can also be solved by noticing it’s related to the longest common subsequence (LCS) between the string and its reverse. The length of LPS is LCS(s, reverse(s)). You can use that as an alternative approach (DP for LCS).
- For palindromic substring, expanding around each index (consider each index and between each pair of indices as center) yields O(n²) time but in practice is straightforward and efficient.
- When implementing 2D DP, be careful with indices and iteration order. For LPS, you typically increase the length of the substring in the outer loop (from length 2 to n) so that smaller substrings are solved before larger.
- Use appropriate data types for indices in languages (in JS, just be careful with string indices).

**JavaScript Code Example – Longest Palindromic Subsequence:** We compute LPS length using 2D DP:

```js
function longestPalindromeSubseq(s) {
  const n = s.length;
  if (n === 0) return 0;
  // dp[i][j] = length of LPS in s[i..j]
  let dp = Array.from({length: n}, () => Array(n).fill(0));
  // substrings of length 1 are palindromes of length 1
  for (let i = 0; i < n; i++) dp[i][i] = 1;
  // consider substrings of increasing length
  for (let len = 2; len <= n; len++) {
    for (let i = 0; i <= n - len; i++) {
      let j = i + len - 1;
      if (s[i] === s[j]) {
        dp[i][j] = 2 + (i+1 <= j-1 ? dp[i+1][j-1] : 0);
      } else {
        dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
      }
    }
  }
  return dp[0][n-1];
}
```

*Comments:* We fill the table by increasing substring length. When `s[i] == s[j]`, they can form a palindrome’s two ends, so we add 2 to the LPS of the inside substring `i+1..j-1`. If `i+1 > j-1` (meaning the substring length is 2 or 1), we add 2 (or 1 in case they overlap, but our loop covers base case). If they don’t match, we take the max of dropping either the left or right character. The answer is `dp[0][n-1]`, the LPS of the whole string ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=,Output)).

**JavaScript Code Example – Longest Palindromic Substring:** We expand around center for brevity (DP table approach omitted for time complexity reasons):

```js
function longestPalindrome(s) {
  const n = s.length;
  if (n === 0) return "";
  let start = 0, maxLen = 1;
  for (let center = 0; center < n; center++) {
    // odd length palindromes (single center)
    let l = center, r = center;
    while (l >= 0 && r < n && s[l] === s[r]) {
      if (r - l + 1 > maxLen) {
        start = l;
        maxLen = r - l + 1;
      }
      l--; r++;
    }
    // even length palindromes (double center)
    l = center; r = center + 1;
    while (l >= 0 && r < n && s[l] === s[r]) {
      if (r - l + 1 > maxLen) {
        start = l;
        maxLen = r - l + 1;
      }
      l--; r++;
    }
  }
  return s.substring(start, start + maxLen);
}
```

*Comments:* We consider each index as a potential center of a palindrome. We expand outwards for odd-length palindromes (center at a letter) and even-length (center between two letters). We track the longest found. This is O(n²) in worst case (e.g. “aaaaa…”) but efficient and simpler than a DP table for substrings.

**Problem List (LeetCode examples from easy to hard):**
1. **Longest Palindromic Substring** – *LeetCode 5 (Medium)* – Find the longest palindromic contiguous substring (expand around center or DP table).
2. **Longest Palindromic Subsequence** – *LeetCode 516 (Medium)* – Find length of longest palindromic subsequence ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=,Output)).
3. **Palindromic Substrings** – *LeetCode 647 (Medium)* – Count all palindromic substrings in a string (can do DP or center-expansion to count every palindrome).
4. **Palindrome Partitioning** – *LeetCode 131 (Medium)* – List all ways to partition a string into palindromes (uses DFS with palindrome DP or backtracking).
5. **Palindrome Partitioning II** – *LeetCode 132 (Hard)* – Minimum cuts to partition string into palindromes (DP: use palindrome table + cut DP).
6. **Count Different Palindromic Subsequences** – *LeetCode 730 (Hard)* – Count distinct palindromic subsequences in a string (advanced DP with modulus, handling overlapping substructures carefully).
7. **Longest Palindromic Subsequence II** – variations where you might have to construct the subsequence, or with some constraint.
8. **Palindromic Subsequences in a Range** – queries for LPS in substrings (could precompute DP table for the string).
9. **Longest Bitonic Subsequence** – though not palindromic, sometimes solved with reverse comparisons (LIS in forward and reverse).
10. **Longest Common Palindromic Subsequence** – find LPS that is common to two strings (combines LCS and palindrome logic – not common in interviews).
11. **Palindromic Suffixes** – find longest palindrome starting at some index, etc. (KMP or DP).
12. **Construct Palindrome from Subsequence** – given the LPS length, sometimes reconstruct the subsequence (use DP table and backtrack).
13. **Minimum Insertions to Form Palindrome** – *LeetCode 1312 (Medium)* – Minimum insertions to make string a palindrome (related to LPS: answer = n - LPS length).
14. **Minimum Deletions to Form Palindrome** – *LeetCode 1312 (same as above, essentially) or 516 variant* – Also n - LPS.
15. **Longest Palindromic Subsequence in a matrix** – not typical.
16. **Palindromic Tree (EERTREE)** – specialized structure to handle palindic substrings efficiently (beyond scope of most interviews).
17. **Palindrome Pair Concatenation** – *LeetCode 336 (Hard)* – Find pairs of words that form palindrome when concatenated (trie-based, not DP).
18. **Magic Squares or Palindrome Matrix** – not directly applicable as DP pattern.

## Grid-Based DP

**Summary:** Grid-based DP problems involve computing some result on a 2D grid or matrix by dynamic programming, often moving in the grid. Common tasks include computing paths, minimum costs, or maximum values on a grid with allowed movements (usually right/down or in four directions), or counting ways to reach a cell. These are essentially extensions of 1D DP to two dimensions, where optimal substructure arises from considering the cell’s neighbors (usually top or left neighbors for a grid that you traverse from top-left).

Classic examples:
- **Unique Paths** (LC 62) – count paths from top-left to bottom-right moving only down or right ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)).
- **Minimum Path Sum** (LC 64) – find path with minimum sum of values from top-left to bottom-right.
- **Coin Grid or Obstacle Grid** – variants with obstacles (LC 63) or collecting maximum coins.
- **Edit Distance and LCS** can be seen as grid DP where one string is along one dimension and one along the other (we cover those separately under string alignment).
- **DP on a matrix of states** – e.g. *Dungeon Game* (LC 174) where you compute needed health backwards in a grid.
- **Bomb enemy / other cell DP** – e.g. computing some info for each cell based on entire row/col (could use DP in each direction but not quite same category).

In these problems, `dp[i][j]` often represents some quantity at cell (i,j) computed from `dp[i-1][j]` (top) and `dp[i][j-1]` (left) or other neighboring states. The state transitions mirror moving through the grid.

**Common Techniques and Tricks:**
- Often you can do in-place modification of the input grid to save space (if it’s okay to overwrite it) since you only need values from top/left or bottom/right depending on direction.
- Mind the initialization of the first row and first column, since those often have only one way to propagate (only from left for first row, only from top for first col) ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)).
- If movement is allowed in more directions or with obstacles, carefully manage conditions (e.g. if obstacle, dp = 0 ways or skip in min).
- For problems like Dungeon Game where we compute bottom-up (from bottom-right to top-left), make sure to iterate in the correct order (reverse) and initialize the destination properly.
- Many grid DP problems are solvable with BFS as well if it’s an unweighted grid for reachability, but DP provides a direct approach when the problem is structured (like acyclic movement to right/down).
- Use 2D loops effectively; sometimes easier to use one loop nested if both dimensions need to be iterated.

**JavaScript Code Example – Minimum Path Sum:** Find the path from top-left to bottom-right with minimum sum of cell values:

```js
function minPathSum(grid) {
  let m = grid.length, n = grid[0].length;
  // in-place DP: grid[i][j] will store min path sum to reach (i,j)
  for (let i = 1; i < m; i++) {
    grid[i][0] += grid[i-1][0];  // first column accumulative sum
  }
  for (let j = 1; j < n; j++) {
    grid[0][j] += grid[0][j-1];  // first row accumulative sum
  }
  for (let i = 1; i < m; i++) {
    for (let j = 1; j < n; j++) {
      grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
    }
  }
  return grid[m-1][n-1];
}
```

*Comments:* We initialize the top row and left column by cumulative sums (only one way to reach those). Then each cell takes the min of top or left neighbor plus its own value ([The Ultimate Guide to Dynamic Programming | by Aleks | Medium](https://medium.com/@al.eks/the-ultimate-guide-to-dynamic-programming-65865ef7ec5b#:~:text=In%20the%20Unique%20Paths%20problem%2C,1)). We modify `grid` in place to save space. The answer is at bottom-right.

**Problem List (LeetCode examples from easy to hard):**
1. **Unique Paths** – *LeetCode 62 (Medium)* – Count paths in grid (shown earlier).
2. **Unique Paths II** – *LeetCode 63 (Medium)* – Paths in grid with obstacles (set `dp[i][j]=0` if obstacle, and only add from top/left that are valid).
3. **Minimum Path Sum** – *LeetCode 64 (Medium)* – Minimum sum path in grid (shown above).
4. **Dungeon Game** – *LeetCode 174 (Hard)* – Compute minimum initial health to survive from top-left to bottom-right (solve backward from end).
5. **Cherry Pickup** – *LeetCode 741 (Hard)* – Two traversals in a grid to collect cherries (can be transformed to DP with two agents moving, or DP in a combined state of two positions).
6. **Bomb Enemy** – *LeetCode 361 (Medium)* – Not exactly classic DP (compute for each cell how many enemies can be hit by a bomb in that cell, using DP to accumulate row/column counts).
7. **Maximal Square** – *LeetCode 221 (Medium)* – Find largest 1’s square in a binary matrix (DP: `dp[i][j] = min(dp[top], dp[left], dp[top-left]) + 1` if cell is ’1’).
8. **Maximal Rectangle** – *LeetCode 85 (Hard)* – Largest rectangle of 1’s in a binary matrix (solve by DP of histogram heights + stack, not typical grid DP but uses DP for heights).
9. **Coin Grid Path** – e.g., number of ways to collect coins with right/down moves (combine Unique Paths and coin collection).
10. **Out of Boundary Paths** – *LeetCode 576 (Medium)* – Count paths of length k that move out of grid (DP with an extra dimension for steps).
11. **Path with Maximum Gold** – *LeetCode 1219 (Medium)* – DFS/backtracking problem (no DP since cycles possible and path length limited by backtracking).
12. **Robot in a Grid (CCI)** – classic: count or find a path with obstacles.
13. **Minimum Falling Path Sum** – *LeetCode 931 (Medium)* – Min path sum from any cell in top row to bottom row (can move straight or diagonal down). Simple DP row by row.
14. **Diagonal or Knight moves DP** – not as common, usually use BFS for knight.
15. **Spreadsheet DP** – e.g., if a cell’s value depends on others (like Excel formulas), topologically sort and evaluate (beyond standard grid path).
16. **Tilings on Grid** – *LeetCode 790 and 711 etc* – ways to tile grids with dominos or trominoes (often solved with DP using bitmask for state of each row).
17. **3D DP for Grid** – e.g. *Unique Paths III* – *LeetCode 980 (Hard)* – path that visits all empty cells exactly once (backtracking; can be solved with bitmask + DP since grid small).
18. **Minimum Obstacles Removal to Reach Corner** – *LeetCode 2290 (Hard)* – BFS with 0-1 weights rather than DP.
19. **Grid Edit Distance** – editing one grid to another – uncommon.
20. **Dynamic Maze** – moving in grid with DP, usually BFS is used instead.

## String Alignment (Edit Distance, LCS, etc.)

**Summary:** String alignment problems involve comparing two sequences (often two strings) and finding some measure of similarity or difference. The classic examples are **Edit Distance (Levenshtein Distance)** and **Longest Common Subsequence (LCS)**. These are solved with 2D DP where one dimension corresponds to one string (length m) and the other to another string (length n). Typically, `dp[i][j]` represents the answer for the first i characters of one and first j characters of the other. The DP transitions consider operations like insert/delete/replace for edit distance, or match/mismatch for LCS.

Problems in this category:
- **Edit Distance** (min number of insertions, deletions, substitutions to convert string A to B) ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=What%20is%20Dynamic%20Programming%20)) ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)).
- **Longest Common Subsequence** (length of longest subsequence present in both strings).
- **Longest Common Substring** (longest contiguous substring present in both – similar to LCS but different DP condition).
- **Sequence alignment** (with scoring, like DNA sequence alignment – a generalized edit distance with custom costs).
- **Minimum Insertions/Deletions to transform string** (these reduce to edit distance or related to LCS).
- **Wildcard Matching** (string pattern matching with `?` and `*` wildcards – can use DP similar to edit distance where `*` can match multiple chars).
- **Regular Expression Matching** (DP where state is positions in text and pattern).
- **Interleaving String** (check if one string is interleaving of two others – 2D DP where each dimension is one of the two strings).

**Common Techniques and Tricks:** 
- Most of these use a 2D DP of size (m+1) x (n+1). Initialize base cases for empty string alignments: e.g., edit distance dp[i][0] = i (delete all i chars), dp[0][j] = j ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)); for LCS, dp[0][j] = 0 or dp[i][0] = 0 (empty string against another has 0 common subseq).
- Pay attention to indexing (i or j off-by-one if using 1-indexed DP array).
- For Edit Distance, there are typically three operations: replace (or match if same char), insert, delete. The recurrence: if `A[i] == B[j]` then `dp[i][j] = dp[i-1][j-1]` (no cost for this char, just carry over) ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)); if not, consider min of (dp[i-1][j-1] + replace, dp[i-1][j] + delete, dp[i][j-1] + insert) ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)).
- LCS recurrence: if `A[i] == B[j]`, then `dp[i][j] = 1 + dp[i-1][j-1]`; else `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.
- These can be done in O(m*n) time. Space can be optimized to O(min(m, n)) for edit distance using rolling arrays, since dp[i][j] relies only on previous row and current row entries. But careful for algorithms like LCS which need two rows as well.
- For regex or wildcard matching, states include pattern index and string index, transitions depend on pattern char (literal, `?`, or `*`).
- For interleaving string (LC 97), state (i,j) meaning first i of s1 and first j of s2 can form first i+j of s3.
- Many of these DPs fill naturally row by row or column by column. For edit distance and LCS, any order (row-major or column-major) works as long as you have the dependencies computed (which are top/left/diagonal cells).
- Reconstructing solution: For LCS, you can backtrack from dp[m][n] to get the subsequence. For edit distance, backtrack to output actual edit operations.

**JavaScript Code Example – Edit Distance (Levenshtein Distance):** Compute the minimum edits to convert string `word1` to `word2`:

```js
function minDistance(word1, word2) {
  const m = word1.length, n = word2.length;
  let dp = Array.from({length: m+1}, () => Array(n+1).fill(0));
  // Base cases: transform empty prefix to other
  for (let i = 0; i <= m; i++) dp[i][0] = i;       // i deletions
  for (let j = 0; j <= n; j++) dp[0][j] = j;       // j insertions
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (word1[i-1] === word2[j-1]) {
        dp[i][j] = dp[i-1][j-1];  // no edit needed for this char
      } else {
        dp[i][j] = 1 + Math.min(
          dp[i-1][j-1],  // replace
          dp[i-1][j],    // delete word1[i-1]
          dp[i][j-1]     // insert word2[j-1]
        );
      }
    }
  }
  return dp[m][n];
}
```

*Comments:* We use (m+1)x(n+1) DP table where `dp[i][j]` is edit distance between `word1[0..i-1]` and `word2[0..j-1]`. We initialize first row/col as described ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)). Then fill row by row. If characters match, no new edit needed, take diagonal value. If not, consider the minimum of the three possible edits. The final answer is `dp[m][n]`. This is O(m*n) time and space. (We can optimize to two rows space if needed.)

**Problem List (LeetCode examples from easy to hard):**
1. **Longest Common Subsequence** – *LeetCode 1143 (Medium)* – Classic LCS DP between two strings.
2. **Edit Distance** – *LeetCode 72 (Hard)* – Classic edit distance (shown above).
3. **Longest Common Substring** – (variant of LCS, can be solved with DP or suffix structures; LC doesn’t have a direct problem, but common interview question).
4. **Interleaving String** – *LeetCode 97 (Hard)* – Check if a string is interleaving of two others (2D DP: `dp[i][j]` = true if s3[0..i+j-1] can be formed, depends on `dp[i-1][j]` and `dp[i][j-1]`).
5. **Distinct Subsequences** – *LeetCode 115 (Hard)* – Count how many subsequences of s (source) equal t (target) – DP where `dp[i][j]` = count of ways for s[0..i-1] to form t[0..j-1].
6. **Wildcard Matching** – *LeetCode 44 (Hard)* – String pattern matching with `?` and `*` wildcards (2D DP or 1D optimized).
7. **Regular Expression Matching** – *LeetCode 10 (Hard)* – Pattern with `.` and `*` (DP with careful handling of `*` which can match zero or more of preceding element).
8. **Delete Operation for Two Strings** – *LeetCode 583 (Medium)* – Minimum deletions to make two strings equal. (Can solve by finding LCS and computing deletions = m + n - 2*LCS_length).
9. **Minimum ASCII Delete Sum for Two Strings** – *LeetCode 712 (Medium)* – Similar to above but cost is ASCII sum of deleted characters (DP variant of edit distance where “delete” cost is char code, “match” cost 0, no substitution – essentially a weighted LCS complement problem).
10. **One Edit Distance** – *LeetCode 161 (Medium)* – Check if two strings are at most one edit apart (can be solved greedily or by checking edit distance == 1).
11. **Longest Palindromic Subsequence** – *LeetCode 516 (Medium)* – (This we did in palindromic section; note it’s also LCS of string and reverse).
12. **Sequence Alignment (General)** – e.g. with scoring matrix (not typical in coding interviews).
13. **Shortest Common Supersequence** – *LeetCode 1092 (Hard)* – Find the shortest string that has both given strings as subsequences (related to LCS: length = m + n - LCS, and construction requires merging).
14. **Palindrome subsequence minimum deletion** – *LeetCode 1312 (Medium)* – (Related to edit distance: min deletions to make palindrome = n - LPS).
15. **Scramble String** – *LeetCode 87 (Hard)* – Determine if one string is a scramble of another (uses recursion + memo, not exactly alignment but 3D DP).
16. **Uncrossed Lines** – *LeetCode 1035 (Medium)* – Essentially LCS problem disguised as matching pairs of numbers.
17. **Swap Distance** – (Not standard, maybe minimum adjacent swaps to transform string A to B given same multiset of chars – can be solved by greedy or something else).
18. **Edit Distance with cost for each operation** – (Variation where replace might have different cost or skip cost, solved similarly with weights).
19. **Optimal BST** – not exactly string, but similar DP structure (matrix chain type).
20. **Matrix Chain Multiplication** – *LeetCode 312 (Hard, Burst Balloons)* – Interval DP (different category, but uses 2D DP concept similarly).

## Interval DP (Matrix Chain Multiplication, Burst Balloons, etc.)

**Summary:** Interval DP involves problems where the solution is computed over an interval or segment of the input, and we decide to partition that interval into smaller parts. These often involve choosing a breakpoint *k* between i and j, and solving subintervals [i,k] and [k+1,j]. Classic examples:
- **Matrix Chain Multiplication**: Given an order to multiply matrices, find the parenthesization that minimizes cost. DP on interval [i..j] (i..j are indices of matrices), and we try every partition k between i and j to compute `dp[i][j] = min(dp[i][k] + dp[k+1][j] + cost of multiplying result of those two parts)` ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=Example%3A%C2%A0The%20Shortest%20Path%20problem%20has,the%20following%20optimal%20substructure%20property)).
- **Burst Balloons** (LC 312): Pick an order to burst balloons to maximize coins – can be solved by choosing the last balloon to burst in an interval.
- **Minimum Cost to Merge Stones** or merging problems: where merging segments has a cost and we want min cost (often partition DP).
- **Optimal Binary Search Tree** (classic DP, not LC) – choose a root to minimize search cost (interval partition).
- **Game strategies** sometimes fall here (though often game DP is separate state machine). E.g., *Predict the Winner* (LC 486) can be solved with interval DP treating [i..j] as remaining numbers and computing max score difference.

The hallmark of interval DP is a double loop over interval length and a loop over a partition index within the interval. These are typically O(n³) if naively implemented (because for each interval of length L (O(n)), you try all breakpoints (O(n)) for all starting positions (O(n))). For moderate n (like n <= 200 or 300), this may be fine. Some interval problems can be optimized with clever monotonicity observations or using prefix sums for cost.

**Common Techniques and Tricks:**
- Use appropriate interval boundaries. Often we use 1-indexing for convenience or add sentinel values. For example, in Burst Balloons, it’s common to add 1s to both ends of the array to handle edges easily.
- Iterate length from small to large. Similar to palindromic subsequence DP, ensure that when computing dp[i][j], all smaller subintervals are already computed.
- For each interval [i..j], iterate a split index k from i to j (or i to j-1 as a partition point) and combine results.
- Sometimes the recurrence might need handling of an outside contribution. E.g., in matrix chain, cost = dp[i][k] + dp[k+1][j] + (dimension_i * dimension_{k+1} * dimension_{j+1}) if we have matrix dimensions array.
- Memoization (top-down) can also be used for interval DP to avoid writing triple nested loops, but either way complexity is similar.
- Identify if problem naturally splits by picking something as “last” or “first” in the interval. Often picking the *last operation* to consider yields easier combination of subproblems (like last balloon to burst, last matrix multiplication operation, etc.).

**JavaScript Code Example – Matrix Chain Multiplication:** Given an array `dim` of matrix dimensions where matrix i is `dim[i-1] x dim[i]`, find minimum cost of multiplying all matrices in order:

```js
function matrixChainOrder(dim) {
  let n = dim.length - 1;  // number of matrices
  // dp[i][j] = min cost (scalar multiplications) to multiply matrices i..j (1-indexed for matrices)
  let dp = Array.from({length: n+1}, () => Array(n+1).fill(0));
  // cost is 0 for single matrix i..i
  for (let len = 2; len <= n; len++) {
    for (let i = 1; i <= n - len + 1; i++) {
      let j = i + len - 1;
      dp[i][j] = Infinity;
      for (let k = i; k < j; k++) {
        const cost = dp[i][k] + dp[k+1][j] + dim[i-1] * dim[k] * dim[j];
        if (cost < dp[i][j]) {
          dp[i][j] = cost;
        }
      }
    }
  }
  return dp[1][n];
}
```

*Comments:* We use 1-indexing for matrices 1..n. `dim` has length n+1. The triple loop: `len` is the chain length from 2 to n, `i` is start, `j = i+len-1` is end. Then `k` splits the chain into [i..k] and [k+1..j]. The cost of splitting at k is cost of left part + cost of right part + cost of multiplying the two resulting matrices (dimensions `dim[i-1] x dim[k]` and `dim[k] x dim[j]`, so multiplication cost = `dim[i-1]*dim[k]*dim[j]`). We take the minimum cost ([Optimal Substructure Property in Dynamic Programming | DP-2 - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-substructure-property-in-dynamic-programming-dp-2/#:~:text=Example%3A%C2%A0The%20Shortest%20Path%20problem%20has,the%20following%20optimal%20substructure%20property)). This runs in O(n³).

**Problem List (LeetCode examples from easy to hard):**
1. **Burst Balloons** – *LeetCode 312 (Hard)* – Classic interval DP: choose an interval and last balloon to burst.
2. **Matrix Chain Multiplication** – *Not on LeetCode, classic* – Illustrated above.
3. **Minimum Cost to Merge Stones** – *LeetCode 1000 (Hard)* – Merge adjacent stones with cost = sum, can only merge k at a time (interval DP with an extra dimension for the number of piles mod (k-1)).
4. **Minimum Score Triangulation of Polygon** – *LeetCode 1039 (Medium)* – Given an n-gon, choose order of triangulation to minimize score (DP similar to matrix chain: pick a point to form triangle with ends).
5. **Predict the Winner** – *LeetCode 486 (Medium)* – Game where two players pick ends of an array – can be solved by interval DP computing the score difference the current player can achieve on interval [i..j].
6. **Optimal Binary Search Tree** – *CLRS/GFG classic (Hard)* – Given frequency of keys, build BST with minimal search cost (interval DP choosing root).
7. **Stone Game series** – *LeetCode 877 (Medium), 1140, 1406, 1563 (Hard)* – Various two-player games on intervals of piles (often solved with interval DP or DFS+memo computing score differences).
8. **Palindrome Partitioning II** – *LeetCode 132 (Hard)* – Min cuts for palindrome partition (can be solved via interval DP checking each possible partition point, though usually done with 1D DP with palindrome check).
9. **Egg Dropping** – *Classic (Hard)* – Min trials to find threshold floor (DP interval-like: choose a floor to drop – splits problem into two intervals: below or above).
10. **Burst Balloons II** – (Hypothetical variations, e.g. maximize sum with different formulas).
11. **Merging Stones (K-way)** – variants of merging cost problems.
12. **Matrix Split** – any problem of dividing an array into segments to optimize some cost (if cost of segment can be computed easily, sometimes can use interval DP or 1D DP).
13. **Boolean Parenthesization** – *GFG classic* – Ways to parenthesize boolean expression to get true (interval DP counting ways for each interval to be true/false).
14. **Scramble String** – *LeetCode 87 (Hard)* – Determine if string is scramble of another (interval splitting problem on strings, solved with recursion + memo, effectively interval DP on string).
15. **Minimum Difficulty of Job Schedule** – *LeetCode 1335 (Hard)* – Split array of jobs into d segments minimizing difficulty sum (this is segmented DP, can do with interval logic computing cost of each segment).
16. **Guess Number Higher or Lower II** – *LeetCode 375 (Medium)* – Worst-case cost to guess number (like egg dropping simpler) – solved with interval DP by choosing a pivot to guess (like binary search decision DP).
17. **Optimal Strategy for Game** – *GeeksforGeeks classic* – Similar to Predict the Winner, choose ends to maximize sum (can solve with interval DP).
18. **Triangular Matrix Chain** – variations like maximizing cost instead of min.
19. **Matrix Partition** – not common.
20. **Interval Scheduling DP** – usually done greedily or with binary search, but one could do DP by sorting intervals and doing something like dp[i] = max(value[i] + dp[next_non_conflict(i)], dp[i+1]) – that’s 1D DP with binary search, somewhat interval in nature (weighted interval scheduling).

## State Machine DP (Stock Trading Problems, etc.)

**Summary:** State Machine DP is used when a problem can be modeled as a finite set of states and transitions (often based on time steps or sequence positions). A prime example is the series of **stock trading problems** (Best Time to Buy/Sell Stock with various conditions) which can be seen as a state machine: e.g., state = {holding a stock or not holding a stock, maybe in cooldown, or how many transactions left}. We then iterate through days and update states. Each state has a recurrence relation derived from either staying in the same state or transitioning from another state via some event (buy, sell, cooldown, etc.) ([Dynamic Programming State Machine Fundamentals](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachineConcept/Fundamentals#:~:text=,based%20control%20logic)) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=So%20we%20can%20see%20that,we%20have%20three%20states%20here)).

This pattern appears in:
- Stock problems: 
  - *Unlimited transactions* (cooldown or fee or none) – states like “holding” or “not holding” (and maybe “cooldown”).
  - *Limited transactions (k transactions)* – can use multiple states or dimensions for transaction count.
- Problems where you have a machine with a few modes and events trigger mode changes. For example, a DP solution for the “alternating subarray” (wiggle) uses two states: up or down.
- Some scheduling or string problems can also be seen this way (though often solved with simpler DP).
- Another example: *Maximum Product Subarray* uses two states (max_so_far, min_so_far) at each position due to positive/negative flip – that’s like a state machine with two states (highest product ending here, lowest product ending here).
- Generally, if you find yourself writing separate DP arrays for different scenarios that alternate (like `dp_hold[i]` and `dp_sold[i]`), that’s a state machine DP.

**Common Techniques and Tricks:**
- Define clear states and what each represents. Draw the state transition diagram if possible (e.g., for stocks with cooldown: states = {Hold, Sold (cooldown), Rest} and transitions ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=So%20we%20can%20see%20that,we%20have%20three%20states%20here)) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown))).
- Write recurrence for each state: how can we end up in this state today? Either we stayed in the same state from yesterday, or we came from another state via an action (buy, sell, etc.) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=Below%20is%20the%20state%20machine,diagram)).
- For stock problems:
  - State usually split into “have stock” vs “no stock” (and sometimes an intermediate state for cooldown).
  - Example: no cooldown, no fee: `hold[i] = max(hold[i-1], -price[i] + noHold[i-1])`, `noHold[i] = max(noHold[i-1], price[i] + hold[i-1])`.
  - With cooldown: add a state for “sold/cooldown”.
  - With k transactions: can extend state to depend on transaction count (or use 2D DP: `dp[i][j][0/1]` for day i, j transactions used, holding or not).
- For two-state cases, sometimes you can optimize further by writing formula directly (some stock problems have known greedy solutions).
- Multi-step transitions: sometimes easier to unroll into conditional logic rather than a formula.
- Always initialize states properly (day 0 or day -1). For instance, `hold[0] = -price0` if you buy at day0, or `-Infinity` if not allowed to start with a stock, etc.
- Some seemingly complex problems can become simple with state machine DP by focusing on what decisions need to be made at each step.

**JavaScript Code Example – Stock Trading with Cooldown:** (LC 309) We have three states:
- `hold[i]`: max profit on day i if we **hold** a stock at end of day.
- `sold[i]`: max profit on day i if we just **sold** a stock on that day (so next day is cooldown).
- `rest[i]`: max profit on day i if we have no stock and did not sell on day i (we are in rest or cooldown state, ready to buy).

State transitions:
- `hold[i] = max(hold[i-1], rest[i-1] - price[i])` (either continue holding, or buy today (which requires we were in rest state before today) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=So%20we%20can%20see%20that,we%20have%20three%20states%20here))).
- `sold[i] = hold[i-1] + price[i]` (sell today from a previously held stock).
- `rest[i] = max(rest[i-1], sold[i-1])` (if we have no stock today, either we were already resting, or we cooled down from a sale yesterday) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=3)).

```js
function maxProfitCooldown(prices) {
  let n = prices.length;
  if (n === 0) return 0;
  let hold = -Infinity, sold = 0, rest = 0;
  for (let price of prices) {
    let prevHold = hold, prevSold = sold, prevRest = rest;
    hold = Math.max(prevHold, prevRest - price); // buy or keep holding ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=3))
    sold = prevHold + price;                     // sell today ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=If%20we%20are%20at%20hasStock,we%20transition%20to%20noStock%20state))
    rest = Math.max(prevRest, prevSold);         // no action (or cooldown finished) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=If%20we%20are%20at%20hasStock,we%20transition%20to%20noStock%20state))
  }
  return Math.max(sold, rest); // max profit can be in sold (just sold last day) or rest state
}
```

*Comments:* We use variables to track the states day by day. Initially (`hold = -Infinity` (can’t hold stock without buying), `sold = 0` (no sale yet), `rest = 0` (no stock and haven’t done anything)). Then update each day. At the end, the maximum profit will be either in `sold` or `rest` state (having a stock (`hold`) at end means we didn’t sell it, so that profit is not realized and typically not maximum). The transitions used correspond to the state machine reasoning above ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=So%20we%20can%20see%20that,we%20have%20three%20states%20here)) ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=Below%20is%20the%20state%20machine,diagram)). 

**Problem List (LeetCode examples from easy to hard):**
1. **Best Time to Buy and Sell Stock** – *LeetCode 121 (Easy)* – Only one transaction allowed (can be done greedily, but also DP: one state for hold, one for not-hold with at most one transaction).
2. **Best Time to Buy and Sell Stock II** – *LeetCode 122 (Medium)* – Unlimited transactions, no cooldown/fee (greedy works; DP states: hold/no-hold).
3. **Best Time to Buy and Sell Stock with Cooldown** – *LeetCode 309 (Medium)* – As above, 3-state DP ([Stock Trading With Cooldown](https://www.thealgorists.com/Algo/DynamicProgramming/StateMachine/StockTradingWithCooldown#:~:text=So%20we%20can%20see%20that,we%20have%20three%20states%20here)).
4. **Best Time to Buy and Sell Stock with Transaction Fee** – *LeetCode 714 (Medium)* – 2-state DP (hold/no-hold) with fee subtracted on sell.
5. **Best Time to Buy and Sell Stock III** – *LeetCode 123 (Hard)* – At most 2 transactions (can use DP with 2 transaction states or split into two passes).
6. **Best Time to Buy and Sell Stock IV** – *LeetCode 188 (Hard)* – At most k transactions (DP with dimension for transaction count, or optimize using the formula for each k).
7. **Stock Trading Generalization** – any variation can be handled by extending states (e.g., with fee and cooldown combined).
8. **Maximum Product Subarray** – *LeetCode 152 (Medium)* – Can be seen as 2-state DP: `maxProd[i]` and `minProd[i]` (since a negative * negative becomes positive).
9. **Wiggle Subsequence** – *LeetCode 376 (Medium)* – 2-state DP: `up[i]` = length of wiggle subseq ending at i going up, `down[i]` = length ending going down.
10. **Arithmetic Slices II** – *LeetCode 446 (Hard)* – Actually uses dictionary for differences (not a small fixed state machine, more like multiple states keyed by difference).
11. **Automata in string matching** – e.g., regex/wildcard (treated in string alignment, but you can view pattern matching as state machine as well).
12. **Game DP** – Many turn-based games can be states of game configurations, but often solved with minimax or interval DP.
13. **Finite State Compression** – If a problem’s state can be represented as bitmask and transitions, sometimes bitmask DP is used (next section).
14. **Task Scheduler** – *LeetCode 621 (Medium)* – Greedy solution known, but could be framed as states for cooldown etc.
15. **Output-sensitive DP** – maybe not relevant here.

## Bitmask DP (Subset Problems, TSP, etc.)

**Summary:** *Bitmask DP* is a technique to use a bitmask (an integer whose binary representation corresponds to a subset) to represent state in DP. This is commonly used when the state needs to encode a subset of elements (e.g., which cities have been visited in a traveling salesman tour, which tasks have been completed, which people have been assigned jobs). By using a bitmask of length *n*, we can represent any subset of {0,...,n-1} and use DP over these 2^n states. Bitmask DP is practical for *n* up to about 20 (2^20 = 1,048,576 states, which is manageable). Classic use cases:
- **Traveling Salesman Problem (TSP)**: find shortest route visiting all cities. State = (mask, i) meaning subset of cities visited = mask, and current city = i. Transition: from state (mask, i), go to (mask ∪ {j}, j) for any unvisited j, add distance cost. Use DP or DFS+memo to compute min cost for full mask ([Mastering Bitmask Dynamic Programming: A Comprehensive Guide](https://algocademy.com/blog/mastering-bitmask-dynamic-programming-a-comprehensive-guide/#:~:text=Guide%20algocademy,problems%3B%20Assignment%20problems%3B%20Subset)) ([[PDF] Bitmask DP - Activities](https://activities.tjhsst.edu/sct/lectures/1920/2020_3_6_Bitmask_DP.pdf#:~:text=Therefore%2C%20a%20general%20indicator%20to,%28USACO)).
- **Assignments or covering problems**: e.g. assign jobs to persons (mask of assigned jobs, or mask of persons used). 
- **Subset with conditions**: e.g. find subset meeting some property (some can be done with simpler DP or brute force, but bitmask can iterate subsets).
- **Bitmask DP on graphs**: counting Hamiltonian paths, etc.
- **Competitive programming digit DP** (the term "digit DP" is different and covered later).

Bitmask DP often involves iterating over subsets and using bit operations to manipulate masks (like `mask | (1<<j)` to add an element, `mask & ~(1<<j)` to remove, etc.), and using precomputed values for transitions if possible (like distances matrix in TSP).

**Common Techniques and Tricks:** 
- Use an integer for mask (in JS, bit operations work up to 32 bits reliably; for >32 bits, one might use bit sets or split into two 32-bit, but n usually ≤ 20 for feasibility).
- To iterate over all subsets of size n, for mask from 0 to (1<<n)-1 do something.
- To iterate over subsets of a given mask (common trick in some DP optimizations or inclusion-exclusion DP): `for(sub=mask; sub; sub=(sub-1)&mask)`.
- Memoization (top-down) is often easier to implement for TSP or similar than bottom-up, due to needing to iterate subsets and pick next element.
- The state space is 2^n * n for TSP (n * 2^n states), which for n=15 is 15*2^15 ~ 15*32768 = ~491k, and for n=20 is 20*1,048,576 = ~20 million (which might be borderline but in C++ okay; in JS might be slow but possibly manageable with pruning or bit optimizations).
- Use bit DP for problems like: *Smallest Sufficient Team* (LC 1125) – find smallest subset of people that cover all skills (DP by mask of skills).
- Another example: *1641 Count Vowel Strings* could be solved by DP with bitmask state representing last vowel used (though a combinatorial formula is simpler).
- Bitmask DP can sometimes be combined with other patterns (e.g. bitmask + DP on digits for some counting problems, but that’s beyond typical interview scope).

**JavaScript Code Example – Traveling Salesman (bitmask DP via recursion):** 

```js
function tsp(distanceMatrix) {
  const n = distanceMatrix.length;
  const FULL_MASK = (1 << n) - 1;
  // memo[mask][i] = min cost to start from city i with set 'mask' of visited cities (mask includes i)
  let memo = Array.from({length: 1<<n}, () => Array(n).fill(undefined));
  
  function dp(mask, i) {
    if (mask === FULL_MASK) {
      return 0;  // all cities visited, cost 0 to end (or could add distance to return to start if required)
    }
    if (memo[mask][i] !== undefined) return memo[mask][i];
    let minCost = Infinity;
    // try to go to some city j not yet visited
    for (let j = 0; j < n; j++) {
      if (!(mask & (1 << j))) {  // if j is not in mask (not visited)
        let newMask = mask | (1 << j);
        let cost = distanceMatrix[i][j] + dp(newMask, j);
        if (cost < minCost) {
          minCost = cost;
        }
      }
    }
    memo[mask][i] = minCost;
    return minCost;
  }
  
  // try starting from each city as a start (or fix a start to avoid duplicate cycles; in TSP we can fix start=0 for cycle)
  let result = Infinity;
  for (let start = 0; start < n; start++) {
    result = Math.min(result, dp(1<<start, start));
  }
  return result;
}
```

*Comments:* This solves the *path* version of TSP (not necessarily returning to start; if we need cycle, add return cost in base case). We use `mask` to track visited cities and `i` as current city. Initially, we try all possible starting cities (or fix one to reduce symmetry). The recursion tries all unvisited cities next. We memoize by `memo[mask][i]`. If n=15 or 16, this may be slow in JS but demonstrates the idea. For n=10 or 12, it should be fine. 

**Problem List (LeetCode examples from easy to hard):**
1. **Traveling Salesman Problem (TSP)** – *Not on LeetCode, classic* – Find shortest Hamiltonian cycle (bitmask DP is standard for ~15 cities).
2. **Shortest Path Visiting All Nodes** – *LeetCode 847 (Hard)* – Find shortest path that visits all nodes in an unweighted graph (this is essentially TSP on an arbitrary graph, solved with BFS + bitmask or DP).
3. **Smallest Sufficient Team** – *LeetCode 1125 (Hard)* – Find minimum number of people to cover all skills (bitmask of skills as state, DP or DFS to cover all bits) ([Mastering Bitmask Dynamic Programming: A Comprehensive Guide](https://algocademy.com/blog/mastering-bitmask-dynamic-programming-a-comprehensive-guide/#:~:text=Guide%20algocademy,problems%3B%20Assignment%20problems%3B%20Subset)) ([[PDF] Bitmask DP - Activities](https://activities.tjhsst.edu/sct/lectures/1920/2020_3_6_Bitmask_DP.pdf#:~:text=Therefore%2C%20a%20general%20indicator%20to,%28USACO)).
4. **Stepping Numbers** – (Count numbers where adjacent digits differ by 1 – can be done with digit DP or BFS, not exactly bitmask DP).
5. **Bitmask DP on matching problems:** e.g., **Assignment Problem** – assign N jobs to N people with min cost (bitmask DP solution of complexity O(n^2 * 2^n)).
6. **1434 Number of Ways to Wear Different Hats** – *LeetCode 1434 (Hard)* – There are 40 hats, up to 10 people, each person can wear one of some hats, count ways so that all people wear different hats – classic bitmask DP where mask represents which people have been assigned hats ([Mastering Bitmask Dynamic Programming: A Comprehensive Guide](https://algocademy.com/blog/mastering-bitmask-dynamic-programming-a-comprehensive-guide/#:~:text=Guide%20algocademy,problems%3B%20Assignment%20problems%3B%20Subset)) ([[PDF] Bitmask DP - Activities](https://activities.tjhsst.edu/sct/lectures/1920/2020_3_6_Bitmask_DP.pdf#:~:text=Therefore%2C%20a%20general%20indicator%20to,%28USACO)).
7. **Partition to K Equal Sum Subsets** – *LeetCode 698 (Hard)* – Can also be done with bitmask DP: use mask to represent used elements, and recursively try to fill one bucket at a time (though typical solution is DFS + memo).
8. **Hamiltonian Path in DAG** – can solve with bitmask DP, but usually simpler topologically.
9. **Matching Problems** – e.g. *LC 416 variant: pick subsets for multiple groups* – bitmask can enumerate subsets.
10. **1483 Kth Ancestor of Tree Node** – uses binary lifting (bitmask-like concept but not DP).
11. **968 Binary Tree Cameras** – greedy, skip.
12. **1349 Maximum Students Taking Exam** – *LeetCode 1349 (Hard)* – Arrange students in a grid such that no two cheat (bitmask DP row by row, treating each row configuration as a state, ensuring no conflict with previous row).
13. **Farey Sequence or other math** – likely not relevant.
14. **1755 Closest Subsequence Sum** – *LeetCode 1755 (Medium)* – Find subsequence sum closest to goal (n up to 40, often solved by splitting into two 20-sized sets and using meet-in-middle bit enumeration).
15. **1066 Campus Bikes II** – assign bikes to workers (bitmask DP assignment).
16. **Cyberattack problem** – (like given network and initial keys (bits), find if you can reach target keys – somewhat bitmask BFS).
17. **DP on subsets for inclusion-exclusion** – e.g., count of ways to cover all elements with given sets (bitmask to represent covered elements).
18. **1255 Maximum Score Words** – *LeetCode 1255 (Hard)* – Choose a subset of given words such that you don’t exceed given letter counts and maximize score (at most 15 words, can brute force or bitmask DP).
19. **1074 Number of Submatrices That Sum to Target** – not bitmask DP (2D prefix sum).
20. **691 Stickers to Spell Word** – *LeetCode 691 (Hard)* – Minimum stickers to form a word (bitmask or DFS+memo on remaining letters, bitmask can represent which letters of target remain, though state space is large if target length > 15 – typically solved with bit DP for smaller targets).

## Digit DP (Counting numbers with constraints)

**Summary:** *Digit DP* is a technique to count numbers (usually in a range [0, N]) that satisfy certain digit-related constraints by DP over the digits of N. This is common in competitive programming but less so in interviews, except maybe at Google or top companies for hard problems. It involves a state per prefix of the number, typically tracking whether we are currently on a prefix equal to N’s prefix (tight condition) and some property (like sum of digits so far, or last digit used, or a mask of used digits, etc.). 

For example, counting how many numbers ≤ N have no two adjacent digits the same can be done with digit DP: state (pos, prevDigit, tight) – moving left to right. Or counting numbers with a certain sum of digits, etc.

**Key idea:** We build the number digit by digit. A state often includes:
- `pos` (which digit position we’re filling out of total length),
- some property like current sum or last digit or mask of used digits,
- `tight` (whether we’re restricted by the prefix of N or already lower).
We then transition by choosing a digit for the current position and updating states.

**Common Techniques:** 
- Use recursion with memo or iterative DP for each prefix length.
- If counting for range [A, B], do DP up to B and subtract DP up to A-1.
- `tight` mechanism: If tight=1 (meaning so far prefix equals prefix of N), the max digit we can put is the digit of N at this position; if tight=0, we can put 0-9 freely.
- Ensure to handle leading zeros properly (they often need a state or a trick: you can treat that before any nonzero digit, you are in a "leading zero" state which might affect constraints like not counting those zeros in sum or not considering them as "adjacent" digits).
- Digit DP can handle multiple constraints by enlarging state (which can blow up complexity if not careful).

**JavaScript Code Example – Count numbers ≤ N with no adjacent equal digits:** 

```js
function countNoAdjacentSame(N) {
  let digits = String(N).split('').map(Number);
  let memo = {};
  // dp(pos, prevDigit, tight, started) -> count
  function dp(idx, prev, tight, started) {
    if (idx === digits.length) {
      return 1; // reached end, count 1 valid number
    }
    let key = `${idx},${prev},${tight},${started}`;
    if (memo[key] !== undefined) return memo[key];
    let limit = tight ? digits[idx] : 9;
    let res = 0;
    for (let dig = 0; dig <= limit; dig++) {
      if (!started) {
        // haven't started (leading zeros) 
        if (dig === 0) {
          // still leading zero, don't update prev
          res += dp(idx+1, -1, tight && dig===limit, false);
        } else {
          // starting with a nonzero digit
          res += dp(idx+1, dig, tight && dig===limit, true);
        }
      } else {
        // already started number
        if (dig === prev) continue; // skip same adjacent
        res += dp(idx+1, dig, tight && dig===limit, true);
      }
    }
    memo[key] = res;
    return res;
  }
  return dp(0, -1, true, false);
}
```

*Comments:* We treat `prev = -1` and `started=false` for leading zeros (so that we don’t apply adjacency rule when prev is -1 or before starting). Once a nonzero digit is chosen, `started` becomes true and we enforce `dig !== prev`. The `tight` flag limits `dig` to the current digit of N when applicable. This returns count of [0, N]. We would subtract 1 if we want to exclude 0 itself depending on definition (here 0 has no adjacent digits, so it’s fine to count it). The complexity is O(number of digits * 10 * state_of_prev * tight * started) ~ O(len*10*something). Here prev can be 0-9 or -1, so at most 11 possibilities, tight 2, started 2, making state ~ 11*2*2 ~44, times length ~ maybe 20 for a 64-bit N, so ~44*20*10 ≈ 8800 computations, very fast.

**Problem List (conceptual, since few directly on LeetCode):**
1. **Count numbers with certain digit sum** – (Given N, count numbers ≤ N with digit sum = S).
2. **Count numbers without restricted digits** – (e.g., no '4' or no "13" substring, etc).
3. **Count numbers with monotonic digits** – (Increasing or decreasing digits).
4. **Count stepping numbers** – (adjacent digits differ by 1).
5. **Count numbers with unique digits** – *LeetCode 1012 (Hard)* – Count numbers ≤ N with no repeated digits (can do digit DP with mask of used digits in state).
6. **Count numbers divisible by X** – (digit DP for divisibility, tracking remainder mod X in state).
7. **Find k-th smallest number with property** – sometimes can use digit DP + binary search on the count.
8. **Sum of digits in range** – (digit DP can accumulate sum of all digits of numbers in range).
9. **LeetCode 600** – *Non-negative Integers without Consecutive Ones (Hard)* – Count ≤ N with no consecutive 1 bits (this is like a binary digit DP).
10. **LeetCode 902** – *Numbers At Most N Given Digit Set (Hard)* – Count numbers ≤ N formed from a given set of digits (can be done with combinatorics and digit DP).
11. **LeetCode 1012** – as mentioned, no repeated digits.
12. **LeetCode 1397** – *Find all Good Strings (Hard)* – Involves counting strings ≤ given length with forbidden substring (this is like digit DP but on alphabet instead of digits, combined with Aho-corasick automaton for the forbidden substring state).
13. **Project Euler problems** – many can be solved with digit DP (like counting numbers with certain property in a large range).
14. **UVa/ICPC classic DP** – e.g., count palindromic numbers in range, etc.
15. **Count primes in range by digits** – possible but usually use math.
16. **Next lexicographical number with property** – not exactly counting, more search.
17. **Construct smallest/largest number under constraints** – sometimes solved with DP akin to digit DP but with greedy since we want actual number, not count.
18. **Digital DP for probability** – rarely, maybe for random processes counting states.
19. **Binary DP** – like digit DP but in binary (like LC 600).
20. **Anything with “at most N” and “no something in digits”** usually is digit DP.

## Decision Making DP (Minimax / Game Theory)

**Summary:** This pattern is about two-player games or scenarios where decisions depend on an adversary or minimizing a maximum loss. Unlike standard DP which often optimizes a single perspective, here we have typically two perspectives (players) with opposite goals. Dynamic programming (or recursion with memo) can be used to solve these by computing something like the *game state value* assuming optimal play. Common examples:
- **Turn-based games on arrays or intervals:** e.g., take coins from ends (Predict the Winner, Stone Game). We use DP or recursion to compute the difference in score the current player can achieve over the other if both play optimally.
- **Minimax search problems:** can often be memoized. For instance, *Can I Win* (LC 464) uses recursion with bitmask state and simulates players.
- **Games like Tic-tac-toe, etc.:** small games can be solved by DFS with memo of states (bitmask of board).
- **Nim and Grundy’s theorem:** many impartial games reduce to computing nimbers via DP on game states.

Key approach is usually: define `dp[state]` as something like the maximum score difference the current player can achieve from this state (positive means current player is ahead by that much). Then:
- If it’s the current player’s turn, they choose a move that maximizes their outcome (i.e., maximize score difference).
- If it’s the opponent’s turn, and we still compute from current player’s perspective, the opponent will choose a move that *minimizes* the score difference (i.e., current player gets the worst outcome from their choices).

This often yields recurrences like: `dp[state] = max( outcome_of_move_i) ` for current player's turn, and if calculating in one function, the opponent’s turn would be `dp[state] = min( outcome_of_move_i)`, or one can flip signs and always take max.

A common simpler approach in take-away games: compute `dp[i][j]` as the best **relative score** the current player can achieve from subarray i..j. If they take i, they gain `values[i]` then the opponent can achieve `dp[i+1][j]` in the remainder, so net result = `values[i] - dp[i+1][j]` (because whatever the opponent achieves is subtracted from current’s score) ([Optimal Strategy for a Game - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-strategy-for-a-game-dp-31/#:~:text=%3E%20%20%20,game)). Similarly if take j: net = `values[j] - dp[i][j-1]`. Then `dp[i][j] = max of those choices` ([Optimal Strategy for a Game - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-strategy-for-a-game-dp-31/#:~:text=%3E%20%20%20,game)). If the result is positive, current player is better by that much.

**Common Techniques and Tricks:** 
- Often simpler to define dp as the *difference* between scores rather than two separate scores, to avoid a 2D state for whose turn it is. That trick was used above and in “Predict the Winner” etc.
- If state space is small (like small n or bitmask of moves), can do recursion with memo. E.g. “Can I Win” uses bitmask of used numbers as state.
- Recognize impartial games that reduce to nim: if a game can be split into independent components, compute Grundy values for each and XOR. (This is more theory, but sometimes relevant if encountered.)
- In minimax search, memoization (transposition tables in chess etc.) can drastically improve performance by avoiding recalculating states.
- In decision DP, ensure to consider all moves and assume optimal response by opponent.

**JavaScript Code Example – Predict the Winner (coin row game):** Two players choose from either end of an array. We compute if player1 can win (get >= half sum). We use the score-difference DP approach:

```js
function PredictTheWinner(nums) {
  const n = nums.length;
  // dp[i][j] = max score difference current player can achieve from subarray i..j
  let dp = Array.from({length: n}, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    dp[i][i] = nums[i]; // only one element, current player takes it
  }
  for (let len = 2; len <= n; len++) {
    for (let i = 0; i <= n - len; i++) {
      let j = i + len - 1;
      // take left => you get nums[i], opponent can then achieve dp[i+1][j]
      let takeLeft = nums[i] - dp[i+1][j];
      // take right => you get nums[j], opponent then achieves dp[i][j-1]
      let takeRight = nums[j] - dp[i][j-1];
      dp[i][j] = Math.max(takeLeft, takeRight);
    }
  }
  return dp[0][n-1] >= 0;
}
```

*Comments:* If `dp[0][n-1] >= 0`, it means the first player can at least tie (difference 0 or more) ([Optimal Strategy for a Game - GeeksforGeeks](https://www.geeksforgeeks.org/optimal-strategy-for-a-game-dp-31/#:~:text=%3E%20%20%20,game)). This uses the relation: current move value minus result if opponent plays optimally on the remainder. The base case length1 is trivial. This is O(n²). (For actual score of both players, one could derive if needed: totalSum + dp[0][n-1] divided by 2 is first player’s score.)

**Problem List (LeetCode examples from easy to hard):**
1. **Predict the Winner** – *LeetCode 486 (Medium)* – Shown above.
2. **Stone Game** – *LeetCode 877 (Medium)* – Similar to Predict the Winner but guaranteed first player win for even count; still solved with similar DP.
3. **Stone Game II** – *LeetCode 1140 (Medium)* – More complex variation with variable take count (DP with states (i, M) and formula considering opponent’s optimal moves).
4. **Stone Game III** – *LeetCode 1406 (Hard)* – Can take 1-3 stones; determine winner (similar to coin row but 1-3 from one end, solved with 1D DP).
5. **Stone Game IV** – *LeetCode 1510 (Hard)* – Players take a square number of stones from a pile, determine winner (this is a take-away game, solved by DP or by finding winning positions – basically a Grundy).
6. **Nim Game** – *LeetCode 292 (Easy)* – Simple nim (if any pile count not zero and no misère conditions, result based on xor).
7. **Flip Game II** – *LeetCode 294 (Medium)* – Can flip “++” to “--”, determine if starting player can force win (DFS with memo on string states).
8. **Can I Win** – *LeetCode 464 (Medium)* – Pick numbers 1..maxChoosable, reach target sum, determine winner (DFS + bitmask memo for states).
9. **Cat and Mouse** – *LeetCode 913 (Hard)* – Game on a graph with two players and draw conditions (states with positions of cat, mouse, and whose turn; can be solved via BFS or DP of states).
10. **Sudoku or puzzles** – not really DP (more backtracking).
11. **Minesweeper** – simulation/backtracking.
12. **Poker or card DP** – Some problems simulate card draws or game outcomes with DP (less common).
13. **Chess endgames** – Not on LeetCode, but can be modeled as state (like King and Rook vs King mate in few moves – solved via retrograde analysis DP).
14. **Game Theory** – any impartial combinatorial game can be solved by DP computing Grundy numbers if state space small (e.g., take-away games, etc.).
15. **253 Matches to Remove** – (just an example, not real LC).
16. **817 Chalkboard XOR Game** – *LeetCode 810 (Hard)* – A game where players remove numbers, solved via math (if xor=0 or not).
17. **Jump Game (two players)** – not in LC, but you can imagine games derived from puzzles.
18. **Hackenbush / Kayles Nim** – special games can be solved by splitting into components and DP per component.
19. **Misère Nim** – variation for special case when all piles size 1.
20. **Atropos (positional game)** – typically out of scope.

# 3. Final Tips

**Approaching DP problems in interviews:**
- **Identify DP candidacy:** Look for keywords like “maximum”, “minimum”, “number of ways”, or problems that involve making a sequence of decisions optimally. If a brute force solution looks exponential (e.g., try all subsets or all sequences) and the problem has overlapping subproblems, it’s likely a DP. Also check for optimal substructure – if the problem can be broken into independent smaller problems whose optimal solutions compose an optimal solution, that’s a strong sign ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=,be%20solved%20using%20Dynamic%20Programming)).
- **Define the state carefully:** This is the hardest and most important part. Think about what parameters define a subproblem. For a sequence, it might be an index (or two indices for subsequences). For 2D grids, it’s coordinates. For game or combinatorial, it could be a tuple like (i, j, …) or a bitmask. Aim for the *smallest* state that still captures all information needed ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=State%3A)). Write a verbal explanation: “dp[x] = ... meaning of dp[x]”. If you can’t describe it clearly, refine the state.
- **Derive the recurrence:** Once state is defined, imagine you know the answers to smaller states – how to get the answer for the current state? Often it’s considering one decision (like choose an item or not, take a step, split an interval at k, etc.). Write out the formula or recursive relation. Ensure you cover all possibilities (use max/min/sum as appropriate).
- **Handle base cases:** Determine the simplest subproblem values directly (like dp[0] = 0, or dp[i][i] = value[i], etc.). 
- **Top-down vs Bottom-up:** In an interview, explaining in words or pseudocode is fine. Many choose bottom-up (iterative) to avoid recursion pitfalls and because it’s straightforward to show loops. But top-down (memoization) is equally acceptable – sometimes easier to implement if state space is complex. Make sure to mention using memoization to avoid exponential blowup ([Memoization vs Tabulation in DP. What is Dynamic Programming (DP)? | by Aryan Jain | Medium](https://medium.com/@aryan.jain19/memoization-vs-tabulation-in-dp-4ff137da8044#:~:text=recursion,implemented%20by%20memoization%20or%20tabulation)).
- **Optimize space if needed:** Once a correct DP solution is clear, consider if you can reduce space. Many 1D or 2D DPs can be optimized since you might only need the last few states (for example, Fibonacci only needs two previous values ([How would you solve the knapsack problem using dynamic ... - Taro](https://www.jointaro.com/interview-insights/amazon/how-would-you-solve-the-knapsack-problem-using-dynamic-programming-including-the-time-and-space-complexity-analysis-and-optimizations/#:~:text=Taro%20www,on%20the%20previous%20row)), LCS needs previous row, etc.). Mention this if relevant: “We can optimize space by only keeping the last row, because dp[i][j] depends only on dp[i-1][*] and dp[i][*-1]” ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=Speed%20Fast%2C%20as%20we%20do,The%20table%20is%20filled)).
- **Be mindful of time complexity:** Analyze your DP. If your state is (i,j) and each takes constant time, that’s O(n²). If it’s (mask, i) with bitmask of size 2^n, that might be O(n * 2^n). Check n’s size to ensure it’s feasible. If it’s borderline, mention optimizations or that it might pass given constraints.
- **Common DP pitfalls:** One pitfall is **double counting** in counting DP – ensure each combination is counted once (order your loops properly). Another is **invalid states** – e.g., accessing dp for a state that isn’t computed yet (in bottom-up ensure correct loop order) or forgetting to initialize something. Off-by-one errors in indexing are also common – carefully set loop bounds especially when using lengths.
- **Testing small cases:** It helps to manually compute DP for a small input to verify the recurrence. Also consider edge cases like empty input or minimal values.
- **Communicate your reasoning:** In an interview, explain how you recognized it as DP and how you set up the state and recurrence. Use a small example to illustrate your DP table or recursion tree. This shows you understand the process, not just memorized a solution.

**Space optimization techniques:** As mentioned, many DPs don’t require storing the entire table. For example:
- In Fibonacci-style and linear DP, you can often keep just last one or two values instead of an array ([How would you solve the knapsack problem using dynamic ... - Taro](https://www.jointaro.com/interview-insights/amazon/how-would-you-solve-the-knapsack-problem-using-dynamic-programming-including-the-time-and-space-complexity-analysis-and-optimizations/#:~:text=Taro%20www,on%20the%20previous%20row)).
- In knapsack 0/1, one can use a 1D array of size capacity+1 and loop backwards for weights ([Tabulation vs Memoization - GeeksforGeeks](https://www.geeksforgeeks.org/tabulation-vs-memoization/?ref=shm#:~:text=Speed%20Fast%2C%20as%20we%20do,from%20the%20first%20entry%2C%20all)).
- In LCS or edit distance, one can keep only previous row and current row (2 rows of length n+1) instead of the full matrix, because the DP transition looks only one row back.
- Memory-heavy DPs (like bitmask DP for TSP) might need bit-level compression or careful allocation in low-level languages, but in high-level usually it’s okay if within bounds.
- If an answer only asks for an optimal value and not reconstruction, you can often compress space. If you need to reconstruct the solution (path, sequence), you might store predecessor pointers or recompute by tracing comparisons.

**Common DP pitfalls and how to avoid them:**
- *Incorrect subproblem dependencies:* Make sure you’re not using a subproblem that isn’t actually smaller. For example, if you had `dp[i]` depending on `dp[j]` where j > i, that’s not valid for a simple forward loop – that might indicate you need a different iteration order or a different state definition.
- *Overcounting in combinations:* Ensure that if order shouldn’t matter, you don’t count permutations as distinct. For combinatorial problems, often fixing an iteration order (like iterate items outside and sum inside for combinations vs the opposite for permutations) is key to avoid duplicates.
- *Not considering all decisions:* e.g., in interval DP, forgetting to try a particular cut, or in coin change, forgetting one of the two choices (taking coin or not).
- *Memory limit:* If your DP table is extremely large (like 10^5 x 10^5, which is 10^10 entries – impossible), the state needs to be reduced (maybe there’s a mathematical formula or greedy solution instead). Always calculate roughly the DP array size.

**Effective DP problem-solving strategies:**
1. **Start small:** Solve the problem for small inputs by hand. This often reveals the pattern or recurrence.
2. **Define state by what varies:** If input is a sequence, likely an index or two indices vary. If it’s a set or subset, maybe a bitmask or an index and a subset size. If it’s two sequences, maybe two indices (like i, j for LCS).
3. **Think recursively:** If I’m at a certain index or state, what choices do I have and how do they lead to subproblems? Writing a recursive solution first can naturally lead to identifying the DP state and recurrence (just add memoization to that recursion).
4. **Use examples to validate:** Try a simple example and maybe a slightly larger one. Fill a DP table on paper or trace the recursion to ensure the recurrence holds.
5. **Check boundary conditions:** For DP arrays, ensure you initialized dp[0] or dp[..][0] correctly, and that your loops cover the full range.
6. **Practice typical patterns:** It’s useful to practice each of the patterns described above on a couple of problems. Many interview DP problems fall into these categories (paths in grid, knapsack, sequences like LIS or edit distance, game DP, etc.), so recognizing them helps quickly zero in on a solution approach.

By systematically breaking problems into subproblems and carefully reasoning about states and transitions, you can master dynamic programming for interview questions. DP is all about finding that recursive structure and exploiting it. Practice on varied patterns is key – eventually, you’ll see a new problem and think “this feels like a knapsack” or “this looks like an interval DP” and you’ll have a blueprint to start with. Good luck, and happy coding! ([Steps to solve a Dynamic Programming Problem - GeeksforGeeks](https://www.geeksforgeeks.org/solve-dynamic-programming-problem/#:~:text=Steps%20to%20solve%20a%20Dynamic,programming%20problem))
