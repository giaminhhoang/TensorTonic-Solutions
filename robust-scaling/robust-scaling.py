def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    # Write code here
    if len(values) == 1:
        return [0.0]
    
    v_sorted = sorted(values)

    def median(v_sorted):
        n = len(v_sorted)
        return 0.5*(v_sorted[n//2-1] + v_sorted[n//2]) if n % 2 == 0 else v_sorted[n//2]
    v_med = median(v_sorted)
    
    n = len(v_sorted)
    if n % 2 == 0: 
        q1 = median(v_sorted[:n//2])
        q3 = median(v_sorted[n//2:])
    else:
        q1 = median(v_sorted[:n//2])
        q3 = median(v_sorted[n//2+1:])

    if q1 == q3:
        return [v - v_med for v in values]
    else:
        return [(v - v_med) / (q3 - q1) for v in values]
