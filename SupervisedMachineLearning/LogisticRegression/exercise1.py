# UNQ_C1
# GRADED FUNCTION: sigmoid
import math
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    ### START CODE HERE ### 
    sg = lambda n : 1 / (1 + math.pow(math.e,-n))
    
    def dive(n):
        if hasattr(n,'__iter__'):
            return [dive(m) for m in n]
        else:
            return  sg(n)
    
    ### END SOLUTION ###  
    
    return dive(z)