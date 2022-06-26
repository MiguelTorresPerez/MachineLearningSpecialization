# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    
    sft = lambda x : np.exp(x)/np.sum(np.exp(x))
    a = sft(z)

    ### END CODE HERE ### 
    return a