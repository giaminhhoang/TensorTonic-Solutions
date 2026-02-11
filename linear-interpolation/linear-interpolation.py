def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    for i in range(1, len(values)):
        if values[i] is not None:
            continue
        else:
            left = i - 1
            for j in range(i + 1, len(values)):
                if values[j] is not None:
                    right = j
                    slope = (values[right] - values[left]) / (right - left)

                    interpolated = [
                        values[left] + slope * (k - left)
                        for k in range(i, right)
                    ]

                    values[i:right] = interpolated
                    i = right
                    break
                else:
                    continue
    return values
