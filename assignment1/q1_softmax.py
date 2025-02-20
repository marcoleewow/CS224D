
# %load q1_softmax.py
import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """
    nrow = len(x)
    
    if isinstance(x[0], np.ndarray)==True:
        ncol = len(x[0])
    else:
        #1D array case
        c = np.amax(x)
        x -= c
        x = np.exp(x)
        expsum = np.sum(x)
        x /= expsum
        #x = [i/expsum for i in x]

        return x
    
    #2D array case    
    for i in range(nrow):
        c = np.amax(x[i,:])
        x[i] -= c
        
    x = np.exp(x)
    expsum = np.sum(x,axis=1)
    
    for i in range(nrow):
        for j in range(ncol):
            x[i,j] /= expsum[i]
    return x

# def softmax(x):
#     """
#     Compute the softmax function for each row of the input x.

#     It is crucial that this function is optimized for speed because
#     it will be used frequently in later code.
#     You might find numpy functions np.exp, np.sum, np.reshape,
#     np.max, and numpy broadcasting useful for this task. (numpy
#     broadcasting documentation:
#     http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

#     You should also make sure that your code works for one
#     dimensional inputs (treat the vector as a row), you might find
#     it helpful for your later problems.

#     You must implement the optimization in problem 1(a) of the
#     written assignment!
#     """
#     ### YOUR CODE HERE
#     log_c = np.max(x, axis=x.ndim - 1, keepdims=True)
#     #for numerical stability
#     y = np.sum(np.exp(x - log_c), axis=x.ndim - 1, keepdims=True)
#     x = np.exp(x - log_c)/y
#     ### END YOUR CODE
#     return x

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print ("You should verify these results!\n")

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
#     print ("Running your tests...")
    ### YOUR CODE HERE
#     raise NotImplementedError
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
    
                    