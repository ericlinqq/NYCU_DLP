#%%
from re import L
from selectors import EpollSelector
import numpy as np
import matplotlib.pyplot as plt
import math

class DataGenerator :
    def __init__(self, type):
        self.set_type(type)
    
    def set_type(self, type):
        assert(type == 'Linear' or type == 'XOR')
        self.__type = type
    
    def __generate_linear(n):
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1]) / 1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

    def __generate_XOR_easy():
        inputs = []
        labels = []
        
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue
            
            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)
        return np.array(inputs), np.array(labels).reshape(21, 1)

    def get_data(self, n=100):
        func_dict = {
            'Linear': DataGenerator.__generate_linear(n),
            'XOR': DataGenerator.__generate_XOR_easy()
        }
        func = func_dict[self.__type]

        return func
        
def init_parameter(layer_dims):
    np.random.seed(56)
    parameters = {}
    Layers = len(layer_dims)
    
    # Initialize weight matrix
    for l in range(1, Layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])

    return parameters

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def forward_pass(X, parameters):
    w_num = len(parameters) // 2
    A = X.T
    records = []

    for i in range(1, w_num+1):
        linear_record = {}
        activation_record = {}
        A_prev = A
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]

        # Linear calculation
        linear_record['W' + str(i)] = W
        linear_record['b' + str(i)] = b
        linear_record['A_prev' + str(i)] = A_prev
        Z = np.dot(W, A_prev) + b

        # sigmoid calculation
        A = sigmoid(Z)
        activation_record['Z' + str(i)] = Z
        # print(A.shape)

        record = (linear_record, activation_record)
        records.append(record)

    assert(A.shape == (1, X.shape[0]))

    return A, records


def loss(predict, label):
    m = label.shape[0]
    
    ce = (1./m) * (-np.dot(label.T, np.log(predict).T) - np.dot((1-label).T, np.log(1-predict).T))
    return ce

def backward_pass(AL, X, Y, records):
    grads = {}
    L = len(records)
    m = AL.shape[1]
        
    dC_dAL = - (np.divide(Y.T, AL) - np.divide(1-Y.T, 1-AL))
    dC_dZL = dC_dAL * derivative_sigmoid(AL)
    assert(dC_dZL.shape == AL.shape)

    linear_record, activation_record = records[-1]
    A_prev = linear_record['A_prev' + str(L)]
    W = linear_record['W' + str(L)]
    b = linear_record['b' + str(L)]
    m = A_prev.shape[1]
    dC_dW = 1./m * np.dot(dC_dZL, A_prev.T)
    dC_db = 1./m * np.sum(dC_dZL, axis=1, keepdims=True)

    dC_dAprev = np.dot(W.T, dC_dZL)
    assert (dC_dAprev.shape == A_prev.shape)  # check shape
    assert (dC_dW.shape == W.shape)
    assert (dC_db.shape == b.shape)
    
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = dC_dAprev, dC_dW, dC_db
    
    for l in reversed(range(1, L)):
        linear_record, activation_record= records[l-1]
        
        Z = activation_record['Z' + str(l)]
        dC_dZ = np.array(grads["dA" + str(l)]) * derivative_sigmoid(sigmoid(Z))
        assert (dC_dZ.shape == Z.shape)  
        
        A_prev= linear_record['A_prev' + str(l)]
        W = linear_record['W'+ str(l)]
        b = linear_record['b' + str(l)]
        m = A_prev.shape[1]
        dC_dW = 1./m * np.dot(dC_dZ, A_prev.T)
        dC_db = 1./m * np.sum(dC_dZ, axis = 1, keepdims = True)
        dC_dAprev = np.dot(W.T, dC_dZ)

        assert (dC_dAprev.shape == A_prev.shape)
        assert (dC_dW.shape == W.shape)
        assert (dC_db.shape == b.shape)

        grads["dA" + str(l-1)], grads["dW" + str(l)], grads["db" + str(l)] = dC_dAprev, dC_dW, dC_db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate*grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]

    return parameters


def model(X, Y, layer_dims, epoch=3000, learning_rate=0.08, print_step=1000):
    costs = []
    parameters = init_parameter(layer_dims)

    for i in range(epoch):
        AL, records = forward_pass(X, parameters)

        cost = loss(AL, Y)

        grads = backward_pass(AL, X, Y, records)

        parameters = update_parameters(parameters, grads, learning_rate)

        if i % print_step == 0:
            print(f"Epoch: {i} Loss: {cost}\n")

        costs.append(cost)

    return parameters 

def predict(X, Y, parameters):
    m = X.shape[0]
    n = len(parameters) // 2
    pred = np.zeros([1, m])

    AL, records = forward_pass(X, parameters)
    
    for i in range(m):
        if AL[0, i] > 0.5:
            pred[0, i] = 1
        else:
            pred[0, i] = 0

    correct_num = np.sum(pred == Y.T)
    wrong_num = m - correct_num
    accuracy = correct_num / m 
    print(f"Accuracy: {accuracy}\n")
    print(f"Wrong Prediction Count: {wrong_num}\n")

    return AL, pred

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[0][i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
        
    plt.show() 

if __name__ == '__main__':
    x_train, y_train= DataGenerator('Linear').get_data(1000)
    x_test, y_test = DataGenerator('Linear').get_data(100)
    layer_dims = [2, 10, 10, 1]

    print("---Start training---\n")
    parameters = model(x_train, y_train, layer_dims, 10000, 0.1, 1000)
    print("---Training finished---\n")

    print("Train result:\n")
    train_AL, train_pred = predict(x_train, y_train, parameters)
    
    print("---Start testing---\n")    
    print("Test result:\n")
    test_AL, test_pred = predict(x_test, y_test, parameters)
    print("---Testing finished---\n")
    
    print("Test Prediction:\n")
    print(test_AL.reshape(-1, 1))
    show_result(x_train, y_train, train_pred)
    show_result(x_test, y_test, test_pred)
# %%