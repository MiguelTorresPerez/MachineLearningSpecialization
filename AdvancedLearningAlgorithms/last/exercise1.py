# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    
    if len(y) != 0:
     # Your code here to calculate the fraction of edible examples (i.e with value = 1 in y)
        p1 = len(y[y == 1]) / len(y)
        entropy = entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1) if p1 != 0 and p1 != 1 else 0
           
    ### END CODE HERE ###        
    
    return entropy