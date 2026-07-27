def select_indexes_knapsack(indexes_with_gain_and_size, budget):
    if not indexes_with_gain_and_size:
        return []

    sorted_idx = sorted(indexes_with_gain_and_size, key=lambda x: x[0] / x[1], reverse=True)
    selected = []
    used = 0
    for gain, size in sorted_idx:
        if used + size <= budget:
            selected.append((gain, size))
            used += size
    return selected