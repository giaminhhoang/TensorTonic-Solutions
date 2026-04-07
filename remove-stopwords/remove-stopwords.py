def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stopwords = set(stopwords)
    tokens_new = []
    for token in tokens:
        if token not in stopwords:
            tokens_new.append(token)
    return tokens_new