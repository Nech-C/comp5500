from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # Get the number of training examples
    batch_size = X.shape[0]
    num_classes = W.shape[1]
    # print(f"X.shape: {X.shape}, W.shape: {W.shape}, y.shape: {y.shape}")
    # Loop through each training example
    for i in range(batch_size):
        # print(f"X[{i}]: {X[i]}, W: {W}")
        # print(f"X[{i}].shape: {X[i].shape}, W.T.shape: {W.T.shape}")
        scores = X[i].dot(W)
        # print(f"Scores: {scores}")
        scores -= np.max(scores)
        softmax_probs = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-8)
        loss += -np.log(softmax_probs[y[i]] + 1e-8)

        # gradient calculation
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (softmax_probs[j] - 1) * X[i]
            else:
                dW[:, j] += softmax_probs[j] * X[i]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= batch_size
    loss += reg * np.sum(W * W)
    dW /= batch_size
    dW = dW + 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    batch_size = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    softmax_probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Compute loss
    correct_class_probs = softmax_probs[np.arange(batch_size), y]
    loss = -np.sum(np.log(correct_class_probs + 1e-8)) / batch_size
    loss += reg * np.sum(W * W)

    # Compute gradient
    softmax_probs[np.arange(batch_size), y] -= 1
    dW = X.T.dot(softmax_probs) / batch_size
    dW = dW + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


if __name__ == "__main__":
    X = np.array([[.2, .2, 0.2], [.1, .1, 0.1]])
    W = np.array([[-9.0, 9.0], [-9.0, 9.0], [-1, 2]])
    y = np.array([1, 1])
    reg = 0.1
    print(softmax_loss_naive(W, X, y, reg))
    print(softmax_loss_vectorized(W, X, y, reg))
