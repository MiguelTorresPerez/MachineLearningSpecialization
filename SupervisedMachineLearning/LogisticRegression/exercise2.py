# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    
    loss_sum = 0 
    for i in range(m):
        
        z_wb = 0 
        
        for j in range(n):
            z_wb_ij = w[j]*X[i][j]
            z_wb += z_wb_ij 
            
        z_wb += b 
        f_wb = sigmoid(z_wb)
            
        loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_sum += loss
    
    ### END CODE HERE ### 

    return (1 / m) * loss_sum 