import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
        
    ### YOUR CODE HERE: forward propagation
#     print("data shape =", data.shape)
    Z2 = np.dot(data,W1)+b1 # even b1 is of shape (1,H), this subtracts b1 from ALL rows!
#     print("Z2 shape =",Z2.shape)
    A2 = sigmoid(Z2)
#     print("A2 shape =",A2.shape)
    Z3 = np.dot(A2,W2)+b2
#     print("Z3 shape =",Z3.shape)        
    A3 = softmax(Z3)
#     print("A3 shape =",A3.shape)        
    
    cost =-np.sum(np.log(A3) * labels)

    ### YOUR CODE HERE: backward propagation
    deltaZ3 = A3 - labels
#     print("deltaZ3 shape =",deltaZ3.shape)        

    deltaA2 = np.dot(deltaZ3,W2.T)
#     print("deltaA2 shape =",deltaA2.shape)  

    deltaZ2 = deltaA2*sigmoid_grad(A2)
#     print("deltaZ2 shape =",deltaZ2.shape)
    
    gradW2 = np.dot(A2.T,deltaZ3)
#     print("gradW2 shape =",gradW2.shape)
    gradb2 = np.sum(deltaZ3, axis = 0) #sums up all the column
#     print("gradb2 shape =",gradb2.shape)
    
    gradW1 = np.dot(data.T,deltaZ2)
#     print("gradW1 shape =",gradW1.shape)
    gradb1 = np.sum(deltaZ2, axis = 0)
#     print("gradb1 shape =",gradb1.shape)
    ### END YOUR CODE

    
    
#     ### YOUR CODE HERE: forward propagation
#     hidden = sigmoid(data.dot(W1) + b1)     #hidden = A2
#     prediction = softmax(hidden.dot(W2) + b2) #prediction = A3
#     cost = -np.sum(np.log(prediction) * labels)
#     ### END YOUR CODE

#     ### YOUR CODE HERE: backward propagation
#     delta = prediction - labels
#     gradW2 = hidden.T.dot(delta)
#     gradb2 = np.sum(delta, axis = 0)
#     delta = delta.dot(W2.T) * sigmoid_grad(hidden)
#     gradW1 = data.T.dot(delta)
#     gradb1 = np.sum(delta, axis = 0)
#     ### END YOUR CODE
    
    
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE

    
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()