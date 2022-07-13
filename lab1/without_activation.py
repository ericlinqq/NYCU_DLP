#%%
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    """
    Input: the value that output from sigmoid function.
    """
    return np.multiply(x, 1.0 - x)

class DataGenerator :
    def __init__(self, type):
        self.set_type(type)
    
    def set_type(self, type):
        assert(type == 'Linear' or type == 'XOR')
        self.__type = type
    
    @staticmethod
    def generate_linear(n):
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

    @staticmethod
    def generate_XOR_easy():
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
            'Linear': DataGenerator.generate_linear(n),
            'XOR': DataGenerator.generate_XOR_easy()
        }
        func = func_dict[self.__type]

        return func

class NeuralNetwork:
    def __init__(self, layer_dims, epoch=100000, lr=0.1, print_step=10000):
        """
        Args: 
            layer_dims: a list of the dims in each layer (include input and output layer) [2, dim1, dim2, ..., 1].
        """
        self.set_layer_dims(layer_dims)
        self.__epoch = epoch
        self.__lr = lr
        self.__print_step = print_step
        self.__epsilon = 1e-4
    
    def set_layer_dims(self, layer_dims):
        assert isinstance(layer_dims, list)
        self.__layer_dims = layer_dims
    
    def init_parameters(self):
        """
        Returns:
            A dictionary of parameters that contains the following:
                W(l): Weight matrix in each layer (l-th_dim, (l-1)-th_dim) ndarray 
        """
        np.random.seed(56)
        parameters = {}
        Layers = len(self.__layer_dims)
        
        # Initialize weight
        for l in range(1, Layers):
            parameters['W' + str(l)] = np.random.rand(self.__layer_dims[l], self.__layer_dims[l-1]) 
        return parameters
    
    def forward_pass(self, X, parameters):
        """
        Args:
            X: Input data (data_number, 2) ndarray. 
        Returns:
            A: The result of passing input through the whole network. (1, data_num) ndarray
            records: A list of tuple (linear_record, activation_record) in each layer.
                linear_record: Weight, bias and the input before linear calculation.
                activation_record: The input before activation calculation.
        """
        w_num = len(parameters) 
        A = X.T
        records = []

        for i in range(1, w_num+1):
            linear_record = {}
            activation_record = {}
            A_prev = A
            W = parameters['W' + str(i)]

            # Linear calculation
            linear_record['W' + str(i)] = W
            linear_record['A_prev' + str(i)] = A_prev
            Z = np.dot(W, A_prev)

            # Activation calculation
            activation_record['Z' + str(i)] = Z 
            if i == w_num:
                A = sigmoid(Z)
            else:
                A = Z

            record = (linear_record, activation_record)
            records.append(record)
        
        assert(A.shape == (1, X.shape[0]))

        return A, records

    def compute_loss(self, predict, label):
        """
        Args:
            predict: (1, data_num) ndarray
            label: (data_num, 1) ndarray
        Using Cross entropy as loss function.
        """
        m = label.shape[0]
        
        ce = (1./m) * (-np.dot(label.T, np.log(predict+self.__epsilon).T) - np.dot((1-label).T, np.log(1-predict+self.__epsilon).T))
        return ce

    def backward_pass(self, AL, X, Y, records):
        """
        Args:
            AL: The result of passing input through the whole network. (1, data_num) ndarray
            X: Input data (data_num, 2) ndarray
            Y: Ground-Truth of the input data. (data_num, 1) ndarray
            records: A list of tuple (linear_record, activation_record) in each layer.
        Returns:
            grads: A dictionary of dC_dA(l-1), dC_dW(l), dC_db(l) matrices.
                dC_dA(l-1): ((l-1)-th_dim, data_num) ndarray (same as A(l-1))
                dC_dW(l): (l-th_dim, (l-1)th_dim) ndarrray (same as W(l)) 
        """
        grads = {}
        L = len(records)
        m = AL.shape[1]

        ## The L-th layer    
        dC_dAL = - (np.divide(Y.T, AL+self.__epsilon) - np.divide(1-Y.T, 1-AL+self.__epsilon))
        dC_dZL = dC_dAL * derivative_sigmoid(AL)
        assert(dC_dZL.shape == AL.shape)

        linear_record, activation_record = records[-1]
        A_prev = linear_record['A_prev' + str(L)]
        W = linear_record['W' + str(L)]
        m = A_prev.shape[1]
        dC_dW = 1./m * np.dot(dC_dZL, A_prev.T)

        dC_dAprev = np.dot(W.T, dC_dZL)
        assert (dC_dAprev.shape == A_prev.shape) 
        assert (dC_dW.shape == W.shape)
        
        grads['dW' + str(L)] = dC_dW

        # (L-1)th ~ 1st layer
        for l in reversed(range(1, L)):
            linear_record, activation_record= records[l-1]

            A_prev= linear_record['A_prev' + str(l)]
            W = linear_record['W'+ str(l)]
            m = A_prev.shape[1]
            dC_dW = 1./m * np.dot(dC_dAprev, A_prev.T)
            dC_dAprev = np.dot(W.T, dC_dAprev)
            assert (dC_dAprev.shape == A_prev.shape)
            assert (dC_dW.shape == W.shape)

            grads["dW" + str(l)] = dC_dW

        return grads

    def update_parameters(self, parameters, grads):
        L = len(parameters)

        for l in range(L):
            parameters['W' + str(l+1)] -= self.__lr*grads['dW' + str(l+1)]

        return parameters


    def train(self, X, Y):
        """
        Returns:
            parameters: A dictionary of weight matrix and bias vector in each layer.
            costs: A list of loss in each epoch.
        """
        costs = []
        parameters = self.init_parameters()
        m = Y.shape[0]
        pred = np.zeros([1, m])

        for i in range(1, self.__epoch+1):
            AL, records = self.forward_pass(X, parameters)
            
            for j in range(m):
                if AL[0, j] >= 0.5:
                    pred[0, j] = 1
                else:
                    pred[0, j] = 0

            loss = self.compute_loss(AL, Y)

            grads = self.backward_pass(AL, X, Y, records)

            parameters = self.update_parameters(parameters, grads)

            accuracy = np.sum(pred == Y.T) / m

            if i % self.__print_step == 0:
                print(f"Epoch: {i} Loss: {loss[0][0]} Accuracy: {accuracy}\n")

            costs.append(loss[0][0])
            
            if loss < 0.001:
                break

        return parameters, costs 

    def predict(self, X, Y, parameters):
        """
        Returns:
            AL: Output of the Neural Network.
            pred: 0/1 prediction based on the output.
        """
        m = X.shape[0]
        n = len(parameters)
        pred = np.zeros([1, m])

        AL, records = self.forward_pass(X, parameters)
        
        for i in range(m):
            if AL[0, i] >= 0.5:
                pred[0, i] = 1
            else:
                pred[0, i] = 0

        correct_num = np.sum(pred == Y.T)
        wrong_num = m - correct_num
        accuracy = correct_num / m 
        print(f"Accuracy: {accuracy}\n")
        print(f"Wrong Prediction Count: {wrong_num}\n")

        return AL, pred

    @staticmethod
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

    @staticmethod
    def learning_curve(costs): 
        plt.title('Learning Curve', fontsize=18)
        plt.plot([i for i in range(len(costs))], costs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

def main():
    Data = DataGenerator('Linear')
    x_train, y_train= Data.get_data(1000)
    x_test, y_test = Data.get_data(100) 

    Network = NeuralNetwork(layer_dims=[2, 10, 10, 1], epoch=100000, lr=0.1, print_step=5000)
    print("---Start training---\n")
    start = time.time()
    parameters, costs = Network.train(x_train, y_train)
    end = time.time()
    plt.figure(1)
    NeuralNetwork.learning_curve(costs)
    print("---Training finished---\n")
    print(f"Training Time: {end-start}\n")

    print("Train result:\n")
    train_AL, train_pred = Network.predict(x_train, y_train, parameters)
    print("---Start testing---\n")    
    print("Test result:\n")
    test_AL, test_pred = Network.predict(x_test, y_test, parameters)
    print("---Testing finished---\n")
    
    print("Test Prediction:\n")
    print(test_AL.reshape(-1, 1))

    plt.figure(2)
    NeuralNetwork.show_result(x_train, y_train, train_pred)

    plt.figure(3)
    NeuralNetwork.show_result(x_test, y_test, test_pred)
if __name__ == '__main__':
    main()