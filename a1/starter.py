import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    # Your implementation here
    return

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    return

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    # Your implementation here

    return

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    L_d = 0
    for i in range (0, len(x), 1):
        print(1)
    return

def gradCE(W, b, x, y, reg):
    # Your implementation here
    return

def buildGraph(loss="MSE"):
    #Initialize weight and bias tensors
    tf.set_random_seed(421)
    W = tf.get_variable("W", initializer=tf.truncated_normal(shape=(784, 1), stddev=0.5))
    b = tf.get_variable("b", initializer=tf.truncated_normal(shape=[]))

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    reg = tf.placeholder(tf.float32, shape=(1))
    prediction = tf.matmul(x, W) + b
    
    if loss == "MSE":
    # Your implementation
        L = tf.losses.mean_squared_error(y, prediction)

    elif loss == "CE":
    # Your implementation here
        prediction = tf.sigmoid(prediction)
        L = tf.losses.sigmoid_cross_entropy(y, prediction)

