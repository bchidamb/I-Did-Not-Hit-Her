# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair, Bhairav Chidambaram

import numpy as np

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - Vj.T * (Yij - np.asscalar(Ui * Vj)))

def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - Ui.T * (Yij - np.asscalar(Vj * Ui)))

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    
    #print('U[1]', U[1])
    #print('V.T[:,1]', V.T[:,1])
    #print('product', U[1] * V.T[:,1])
    
    reg_term = 0.5 * reg * (np.sum(np.array(U) ** 2) + np.sum(np.array(V) ** 2))
    err_term = 0.5 * np.sum((Yij - np.asscalar(U[i-1] * V.T[:, j-1])) ** 2 
        for i, j, Yij in Y)
    return (reg_term + err_term) / len(Y)


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    
    # randomly initialize U and V
    U = np.matrix(np.random.uniform(-0.5, 0.5, (M, K)))
    V = np.matrix(np.random.uniform(-0.5, 0.5, (N, K)))
    err_0 = get_err(U, V, Y)
    
    # train for at most 300 epochs
    for e in range(max_epochs):
    
        for idx in np.random.permutation(len(Y)):
        
            i, j, Yij = Y[idx]
            
            U[i-1] = U[i-1] - grad_U(U[i-1], Yij, V.T[:, j-1], reg, eta)
            V[j-1] = V[j-1] - grad_V(V[j-1], Yij, U.T[:, i-1], reg, eta)
        
        # fix early stopping criterion based on first loss reduction
        if e == 0:
            err_1 = get_err(U, V, Y)
            err_prev = err_1
        # check if early stopping criterion is true
        else:
            err_curr = get_err(U, V, Y)
            
            if (err_curr - err_prev) / (err_1 - err_0) <= eps:
                break
                
            else:
                err_prev = err_curr
        
    return U, V, err_curr
