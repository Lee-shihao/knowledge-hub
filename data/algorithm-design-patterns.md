# Algorithm Design Patterns

## Sorting Algorithms

### Quick Sort

Divide-and-conquer algorithm with O(n log n) average complexity.

**Key Idea**: Choose a pivot, partition array around it, recursively sort partitions.

```
Best: O(n log n)
Average: O(n log n)
Worst: O(n²) - when pivot is always min/max
Space: O(log n) for recursion
```

**When to use**:
- General purpose sorting
- In-place sorting needed
- Average case performance matters

### Merge Sort

Stable, guaranteed O(n log n) sorting.

**Key Idea**: Divide array in half, sort each half, merge sorted halves.

```
Time: O(n log n) - all cases
Space: O(n) - needs temporary array
Stable: Yes
```

**When to use**:
- Need guaranteed O(n log n)
- Stable sort required
- External sorting (large files)

### Binary Search

Find element in sorted array in O(log n).

```
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

## Data Structures

### Hash Tables

O(1) average insert, delete, lookup.

**Collision Resolution**:
1. **Chaining**: Linked list at each bucket
2. **Open Addressing**: Find next empty slot

**Load Factor**: `α = n/m` (elements/buckets)
- Chaining: α can be > 1
- Open addressing: α < 1

**Rehashing**: When α exceeds threshold, double bucket count.

### Binary Search Trees

O(log n) operations on balanced trees.

**In-order Traversal**: Yields sorted elements

**Balancing**:
- AVL Trees: Strict balance (height diff ≤ 1)
- Red-Black Trees: Relaxed balance, faster inserts

```
        8
       / \
      4   12
     / \  / \
    2  6 10 14
```

### Heaps

Complete binary tree with heap property.

**Max-Heap**: Parent ≥ children
**Min-Heap**: Parent ≤ children

**Operations**:
- Insert: O(log n)
- Extract max/min: O(log n)
- Peek: O(1)

**Applications**:
- Priority queues
- Heap sort
- Top-k problems

### Graphs

**Representations**:
1. **Adjacency Matrix**: O(1) edge lookup, O(V²) space
2. **Adjacency List**: O(V + E) space, O(degree) edge lookup

**Common Traversals**:

**BFS** (shortest path in unweighted):
```
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**DFS** (cycle detection, topological sort):
```
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

## Dynamic Programming

### Key Concepts

1. **Overlapping Subproblems**: Same subproblems solved multiple times
2. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems

### Approaches

**Top-Down (Memoization)**:
```
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

**Bottom-Up (Tabulation)**:
```
def fib(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### Classic Problems

**Longest Common Subsequence (LCS)**:
```
dp[i][j] = length of LCS of s1[0..i] and s2[0..j]

if s1[i] == s2[j]:
    dp[i][j] = dp[i-1][j-1] + 1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**0/1 Knapsack**:
```
dp[i][w] = max value using items 0..i with capacity w

dp[i][w] = max(
    dp[i-1][w],                    # Don't take item i
    dp[i-1][w-weight[i]] + value[i]  # Take item i
)
```

## Greedy Algorithms

Make locally optimal choice at each step.

**When to use**:
- Greedy choice property: Local optimal leads to global optimal
- Optimal substructure

**Examples**:
- Activity selection
- Huffman coding
- Minimum spanning tree (Kruskal, Prim)
- Shortest path (Dijkstra)

**Activity Selection**:
```
def activity_selection(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected
```

## Divide and Conquer

**Pattern**:
1. **Divide**: Split problem into smaller subproblems
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge solutions

**Examples**:
- Merge Sort
- Quick Sort
- Binary Search
- Strassen's Matrix Multiplication
- Karatsuba Multiplication

## Time Complexity Reference

| Complexity | Examples |
|-----------|----------|
| O(1) | Array access, hash lookup |
| O(log n) | Binary search, heap operations |
| O(n) | Linear search, traversal |
| O(n log n) | Merge sort, heap sort |
| O(n²) | Bubble sort, insertion sort |
| O(n³) | Naive matrix multiplication |
| O(2ⁿ) | Subset enumeration |
| O(n!) | Permutation generation |

## Space-Time Tradeoffs

| Problem | More Time | More Space |
|---------|-----------|------------|
| Search | Linear scan O(n) | Hash table O(1) |
| Cache | Recompute | Memoization |
| Sort | Heap sort O(1) space | Merge sort O(n) space |
| Graph | DFS O(V) stack | BFS O(V) queue |

## Problem-Solving Strategies

1. **Understand the problem**: Inputs, outputs, constraints
2. **Consider edge cases**: Empty input, single element, duplicates
3. **Start with brute force**: Guarantees correctness
4. **Look for patterns**: Sorted? Graph? Tree? DP?
5. **Optimize step by step**: Identify bottlenecks
6. **Test thoroughly**: Corner cases, random inputs
