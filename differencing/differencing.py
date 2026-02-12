def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    temp = series.copy()
    for i in range(1, order+1):
        for j in range(len(series)-i):
            temp[j] = temp[j+1] - temp[j]
    return temp[0:len(series)-order]