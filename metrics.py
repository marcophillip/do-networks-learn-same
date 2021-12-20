import numpy as np
import tensorflow as tf

def HSIC(K,L):

    I = np.ones(len(K))
    n = len(K)
    K_ = np.fill_diagonal()

    L_ = np.fill_diagonal()

    p_1 = np.trace(np.dot(K_,L_))

    p_2 = np.matmul(I.T,np.matmul(K,np.matmul(I,np.matmul(I.T,np.matmul(L,I)))))

    p_2 = p_2/((n-1)*(n-2))

    p_3 = 2*np.matmul(I.T,np.matmul(K,np.matmul(L,I)))

    p_3 = p_3/(n-2)

    return (p_1 +p_2+p_3)/(n*(n-3))


def CKA (X,Y):
    X = tf.reshape(X,[-1,tf.shape(X)[1:]])
    Y = tf.reshape(Y,[-1,tf.shape(Y)[1:]])
    
    p_1 = HSIC(np.matmul(X,X.T),np.matmul(Y,Y.T))

    p_2 = HSIC(np.matmul(X,X.T),np.matmul(X,X.T))

    p_3 = HSIC(np.matmul(Y,Y.T),np.matmul(Y,Y.T))

    return p_1/np.sqrt(p_2*p_3)

