# UNQ_C3
# GRADED CELL: model
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ### 
        Dense(120, activation = 'relu'),      
        Dense(40, activation = 'relu'),         
        Dense(6, activation = 'linear')
  
        ### END CODE HERE ### 

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ### 
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),          
    optimizer = tf.keras.optimizers.Adam(0.01)
    
    ### END CODE HERE ### 
)