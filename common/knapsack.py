# common/knapsack.py

def select_indexes_knapsack(indexes_with_gain_and_size, budget):
    """
    Select indexes using a greedy approach + local search improvement.
    This is fast and gives near‑optimal results for typical workloads.

    Parameters:
        indexes_with_gain_and_size: list of (gain, size)
            gain: float (cost reduction)
            size: int (storage in bytes)
        budget: int (max total size)

    Returns:
        list of selected (gain, size) tuples
    """
    if not indexes_with_gain_and_size:
        return []

    # Step 1: sort by gain/size ratio (greedy)
    sorted_items = sorted(indexes_with_gain_and_size, key=lambda x: x[0] / x[1], reverse=True)

    # Step 2: greedy selection
    selected = []
    used = 0
    for gain, size in sorted_items:
        if used + size <= budget:
            selected.append((gain, size))
            used += size

    # Step 3: local improvement – try to swap one selected item with a non‑selected one
    improved = True
    while improved:
        improved = False
        # For each selected item, try to replace it with a better non‑selected item
        for i in range(len(selected)):
            for item in sorted_items:
                if item in selected:
                    continue
                gain_i, size_i = selected[i]
                gain_j, size_j = item
                # If swapping increases total gain and fits in budget
                if gain_j > gain_i and used - size_i + size_j <= budget:
                    selected[i] = (gain_j, size_j)
                    used = used - size_i + size_j
                    improved = True
                    break  # restart loop after a successful swap
            if improved:
                break

    return selected