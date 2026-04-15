def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)
    intersect = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersect/union if union > 0 else 0.0