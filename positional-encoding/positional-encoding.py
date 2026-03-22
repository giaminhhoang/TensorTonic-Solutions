import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len).reshape(-1,1)
    freq = np.arange(np.ceil(d_model / 2))
    pe = np.zeros([seq_len, d_model])
    pe[:, 0::2] = np.sin(pos / base**(2*freq/d_model))
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(pos / base**(2*freq/d_model))
    else:
        pe[:, 1::2] = np.cos(pos / base**(2*freq[:-1]/d_model))
    return pe