# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    
    for i in range(m):
        
        z_wb = 0 
        
        for j in range(n):
            z_wb_ij = w[j]*X[i][j]
            z_wb += z_wb_ij 
            
        z_wb += b 
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for z in range(n):
            dj_dw_ij = (f_wb - y[i])* X[i][z]
            dj_dw[z] += dj_dw_ij
                
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_db, dj_dw