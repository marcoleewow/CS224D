import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    x2 = x*x
    x2_sum = np.sum(x2, axis = 1)
    rowTotalSum = np.sqrt(x2_sum)
    x = np.divide(x,rowTotalSum[:,None])
    
#     y = np.linalg.norm(x,axis=1,keepdims=True) #<--- simpler method
#     x /= y
    return x

def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print (x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in <--- V
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens    <--- U
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
#     ### YOUR CODE HERE: forward propagation
#     print("predicted shape =",predicted.shape)
#     print("outputVectors shape =",outputVectors.shape)
#     Z = np.dot(outputVectors,predicted)
#     y_hat = np.array(softmax(Z)) #<--- np.asarray change it from list to array
#     cost =-np.log(y_hat[target])

#     ### YOUR CODE HERE: backward propagation
#     delta_y_hat = y_hat
#     delta_y_hat[target] -= 1  
#     N = delta_y_hat.shape[0]    #this is the size of |V|,which is the same as Dx and Dy
#     H = predicted.shape[0]      #this is the size of hidden layer H
#     grad = delta_y_hat.reshape((N,1)) * predicted.reshape((1,H)) #this is dJ/dU
#     gradPred = (delta_y_hat.reshape((1,N)).dot(outputVectors)).flatten() #this is dJ/dV
    
    N, D     = outputVectors.shape

    r    = predicted
    prob = softmax(r.dot(outputVectors.T))
    cost = -np.log(prob[target])

    dx   = prob
    dx[target] -= 1.

    grad     = dx.reshape((N,1)) * r.reshape((1,D))
    gradPred = (dx.reshape((1,N)).dot(outputVectors)).flatten()
    
    
    
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    
    ### sample out some negative samples
    indices = [target]
    labels = np.array([1])
    for i in range(K):
        newIndex = dataset.sampleTokenIdx()
        while newIndex == target:
            newIndex = dataset.sampleTokenIdx()  # <--- keep on sampling until newIndex != target.
        indices.append(newIndex)
        labels = np.append(labels,[-1])     # <--- add -1 to labels vector, now we have (1, -1, ..., -1) (-1: k times)
        
    U = outputVectors[indices,:] # <--- pick out target, and the negative samples' output vector (k*1 x N)
    
    ### YOUR CODE HERE: forward propagation
    Z = np.dot(U,predicted)*labels # <--- times 1 if it is target, times -1 if neg sample
    y_hat = sigmoid(Z)

    J = np.log(y_hat)
    cost = -np.sum(J)

    ### YOUR CODE HERE: backward propagation
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)   
    V = predicted.shape[0] # <--- size of vocab

    delta_y_hat = (y_hat-1)*labels    # <--- sigmoid(u*v_c) - 1, times 1 if it is target, -1 if neg sample
    
    gradPred = np.dot(delta_y_hat.reshape((1,K+1)), U).flatten() # <--- dJ/dVc
    
    gradTemp = np.dot(delta_y_hat.reshape((K+1,1)), predicted.reshape(1,V)) # <--- dJ/dUk (also target word too)
    
    for i in range (K+1):
        grad[indices[i]] += gradTemp[i,:] # <--- other grad is 0, only update target and negative sample

    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    c_index = tokens[currentWord] #<--- get current word vector
    v_c = inputVectors[c_index,:] #<--- get kth row from V
    
    cost = 0.0
    gradIn = np.zeros_like(inputVectors)
    gradOut = np.zeros_like(outputVectors)
    
    for i in contextWords: #<--- loop through window
        contextWords_index = tokens[i]
        cost_i, gradPred_i, grad_i = word2vecCostAndGradient(v_c, contextWords_index, outputVectors, dataset)
        cost += cost_i
        gradOut += grad_i
        
        gradIn[c_index,:] += gradPred_i # <--- this is dJ/dVc, so update Vc only!!! stuck here for a long time

    ### END YOUR CODE

    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
        # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    
    c_index = tokens[currentWord]
    one_hot = np.zeros((2*C,len(tokens))) #<--- tokens is a dict, it has no shape! use len
    
    for i, word in enumerate(contextWords):
        one_hot[i, tokens[word]] = 1.      #<--- one_hot array is all the context words one_hot vectors
        
#     for i in contextWords:
#         contextWords_index = tokens[i]
#         one_hot[i, contextWords_index] = 1 
    
    V = np.dot(one_hot, inputVectors) #<--- V is the set of vk, where they are input vectors for context words
    
    h = (1 / (2*C)) * np.sum(V, axis=0) #<--- why? take average?
    
    cost, gradPred, gradOut = word2vecCostAndGradient(h, c_index, outputVectors, dataset)
    
    gradIn = np.zeros(inputVectors.shape)
    for i in contextWords:
        gradIn[tokens[i]] += (1 / (2*C)) * gradPred
    
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in range(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print ("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print ("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print ("\n=== Results ===")
    print (skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print (cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print (cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()