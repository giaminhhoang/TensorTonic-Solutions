import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    L = math.sqrt(6 / (fan_in + fan_out))
    W = [[w * 2 * L - L for w in row] for row in W]
    return W