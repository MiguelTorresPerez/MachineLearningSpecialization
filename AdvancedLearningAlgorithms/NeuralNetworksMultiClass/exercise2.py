# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        Dense(25, activation='relu', name = "L1"),
        Dense(15, activation='relu',  name = "L2"), 
        Dense(10, activation='linear', name = "L3")

        ### END CODE HERE ### 
    ], name = "my_model" 
)