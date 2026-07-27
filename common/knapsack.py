# common/knapsack.py
import math

def select_indexes_knapsack(indexes_with_gain_and_size, budget):
    """
    Exact knapsack solution using dynamic programming.
    Works with sizes rounded to megabytes (MB) for efficiency.

    Parameters:
        indexes_with_gain_and_size: list of (gain, size)
            gain: float (cost reduction)
            size: int (storage in bytes)
        budget: int (max total size in bytes)

    Returns:
        list of selected (gain, size) tuples
    """
    if not indexes_with_gain_and_size:
        return []

    # Round sizes up to megabytes (1 MB = 1_000_000 bytes)
    # This reduces the DP table size drastically while keeping precision.
    MB = 1_000_000
    # Convert budget to MB (integer division)
    capacity_mb = budget // MB

    # Prepare items with sizes in MB (rounded up to avoid underestimating)
    items = []
    for gain, size in indexes_with_gain_and_size:
        size_mb = math.ceil(size / MB)   # round up to be safe
        items.append((gain, size_mb))

    n = len(items)

    # DP table: dp[i][c] = maximum gain achievable with first i items and capacity c (in MB)
    # We use a 1D array to save memory (rolling array)
    dp = [0.0] * (capacity_mb + 1)

    # For each item, update dp from back to front (0/1 knapsack)
    for i in range(n):
        gain, size_mb = items[i]
        # if size_mb > capacity_mb, this item cannot fit at all, skip it
        if size_mb > capacity_mb:
            continue
        for c in range(capacity_mb, size_mb - 1, -1):
            new_gain = dp[c - size_mb] + gain
            if new_gain > dp[c]:
                dp[c] = new_gain

    # Reconstruct the selected items
    selected = []
    remaining = capacity_mb
    # We need to know which items were selected; we can trace back using a separate table of decisions
    # To avoid storing a 2D table, we can store the choices in a 2D boolean table (n x capacity_mb)
    # But with 150*5000 = 750k booleans, it's fine.
    # We'll build a 2D table for reconstruction (or we can store decisions in a list of lists)
    # Simpler: use a 2D DP table for reconstruction (n+1 x capacity_mb+1)
    # Since we want to keep it simple, we'll use a 2D table of floats (n+1 x capacity_mb+1)
    # But that's 151*5001 floats ≈ 755k floats, about 6 MB, acceptable.
    # However, to be memory efficient, we can store decisions in a bytearray.
    # Let's just build the 2D table for clarity.

    # Re‑run DP with 2D table for reconstruction
    dp2 = [[0.0] * (capacity_mb + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        gain, size_mb = items[i-1]
        for c in range(capacity_mb + 1):
            if size_mb <= c:
                dp2[i][c] = max(dp2[i-1][c], dp2[i-1][c - size_mb] + gain)
            else:
                dp2[i][c] = dp2[i-1][c]

    # Trace back
    c = capacity_mb
    for i in range(n, 0, -1):
        if dp2[i][c] != dp2[i-1][c]:
            # item i-1 was taken
            gain, size_mb = items[i-1]
            selected.append((gain, size_mb * MB))  # restore original size in bytes
            c -= size_mb

    return selected