def select_indexes_knapsack(indexes_with_gain_and_size, budget):
    """
    Selects the best set of indexes that fits within the storage budget.
    Uses Branch and Bound to find the exact optimal solution.

    Parameters:
        indexes_with_gain_and_size: list of tuples (gain, size)
            - gain: positive float, the cost reduction provided by this index.
            - size: positive int, the storage space required (in bytes).
        budget: maximum allowed total size (int).

    Returns:
        list of selected (gain, size) tuples that maximize total gain
        without exceeding budget.
    """
    if not indexes_with_gain_and_size:
        return []

    # Sort items by decreasing gain/size ratio (helps the branch and bound prune faster)
    items = sorted(indexes_with_gain_and_size, key=lambda x: x[0] / x[1], reverse=True)
    n = len(items)

    # Extract gains and sizes for faster access
    gains = [g for g, _ in items]
    sizes = [s for _, s in items]

    best_value = 0.0           # best total gain found so far
    best_combination = [False] * n   # which items are selected in the best solution

    # --- Upper bound function (fractional knapsack) ---
    def bound(level, current_value, current_size):
        """
        Computes the maximum possible gain we could obtain from the remaining items
        if we were allowed to take fractions of them.
        This is used to prune the search.
        """
        remaining = budget - current_size
        bound_value = current_value
        i = level
        # take whole items while they fit
        while i < n and remaining >= sizes[i]:
            bound_value += gains[i]
            remaining -= sizes[i]
            i += 1
        # take a fraction of the next item
        if i < n:
            bound_value += gains[i] * (remaining / sizes[i])
        return bound_value

    # --- Depth-first search with pruning ---
    def dfs(level, current_value, current_size, selected):
        nonlocal best_value, best_combination

        # If we exceed budget, stop this branch
        if current_size > budget:
            return

        # If we processed all items, update best if improved
        if level == n:
            if current_value > best_value:
                best_value = current_value
                best_combination = selected[:]
            return

        # Prune: if even the upper bound cannot beat the current best, stop
        if current_value + bound(level, current_value, current_size) <= best_value:
            return

        # Option 1: skip the current item
        dfs(level + 1, current_value, current_size, selected + [False])

        # Option 2: take the current item
        new_size = current_size + sizes[level]
        if new_size <= budget:
            dfs(level + 1, current_value + gains[level], new_size, selected + [True])

    # Start the search
    dfs(0, 0.0, 0, [])

    # Build the list of selected items
    selected_items = []
    for i, taken in enumerate(best_combination):
        if taken:
            selected_items.append(items[i])

    return selected_items