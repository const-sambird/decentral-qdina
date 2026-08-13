def select_indexes_knapsack(indexes_with_gain_and_size, budget, max_visits=50000):
    """
    Selects the best set of indexes that fits within the storage budget.
    Uses an optimized Branch and Bound algorithm with bitmasks and early stopping
    to prevent O(2^n) explosion during reinforcement learning loops.
    """
    if not indexes_with_gain_and_size:
        return []

    # Sort items by decreasing gain/size ratio to maximize pruning efficiency
    items = sorted(indexes_with_gain_and_size, key=lambda x: x[0] / x[1], reverse=True)
    n = len(items)

    # Extract gains and sizes for faster access
    gains = [g for g, _ in items]
    sizes = [s for _, s in items]

    best_value = 0.0
    best_mask = 0
    visits = 0

    def bound(level, current_value, current_size):
        """
        Computes the fractional knapsack upper bound for remaining items.
        """
        remaining = budget - current_size
        bound_value = current_value
        i = level
        while i < n and remaining >= sizes[i]:
            bound_value += gains[i]
            remaining -= sizes[i]
            i += 1
        if i < n:
            bound_value += gains[i] * (remaining / sizes[i])
        return bound_value

    def dfs(level, current_value, current_size, current_mask):
        nonlocal best_value, best_mask, visits

        # 1. Hard stop to prevent the RL step from freezing on degenerate edge cases
        visits += 1
        if visits > max_visits:
            return

        # 2. Record the best configuration found so far instantly
        if current_value > best_value:
            best_value = current_value
            best_mask = current_mask

        # Reached the end of the items or filled the budget
        if level == n or current_size == budget:
            return

        # Pruning: stop exploring if the theoretical maximum cannot beat the current best
        if current_value + bound(level, current_value, current_size) <= best_value:
            return

        # 3. Branch 1: Take the current item (explored FIRST to raise best_value quickly)
        if current_size + sizes[level] <= budget:
            dfs(level + 1, current_value + gains[level], current_size + sizes[level], current_mask | (1 << level))

        # 4. Branch 2: Skip the current item
        dfs(level + 1, current_value, current_size, current_mask)

    # Start the search
    dfs(0, 0.0, 0, 0)

    # Decode the bitmask back into the selected items
    selected_items = []
    for i in range(n):
        if best_mask & (1 << i):
            selected_items.append(items[i])

    return selected_items