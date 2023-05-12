import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Logistic_regression_ls.csv')

x_data = data[['x1','x2']]
y_data = data['label']
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 0)
y_train = y_train.to_frame()
y_test = y_test.to_frame()

class NeuralNetwork:
    def __init__(self, n, ne, no, N = 1, activation = 'sigmoid'):
        self.n = n # n = number of features
        self.ne = ne # ne = number of neurons in the hidden layer
        self.N = N # N = number of layers in hidden layers
        self.no = no# no = number of output layer neurons
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # initializing the weights and baises
        for i in range(self.N+1):
            if i==0:
                # first hidden layer
                self.weights.append(np.random.randn(self.n, self.ne))
                self.biases.append(np.random.randn(1,self.ne))
            elif i==self.N:
                # last layer
                self.weights.append(np.random.randn(self.ne, self.no))
                self.biases.append(np.random.randn(1, self.no))
            else: 
                # hidden layer
                self.weights.append(np.random.randn(self.ne, self.ne))
                self.biases.append(np.random.randn(1, self.ne))
    
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        return self._sigmoid(z)*(1-self._sigmoid(z))
    
    def _tanh(self, z):
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        return 1 - np.tanh(z)**2
      
    def _backward(self,X, y_pred, y_true):
        # m = no. of samples
        m = X.shape[0]
        
        # Compute the dZ and deltas for the output layer
        dZ = y_pred - y_true
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]
        delta_weights[-1] = np.dot(self.layers[-2].T, dZ) / m
        delta_biases[-1] = np.sum(dZ, axis=0) / m
        
        # Backpropagate the dZ through the hidden layers
        for i in range(self.N, 0, -1):
            dZ = np.dot(dZ, self.weights[i].T) * self._sigmoid_derivative(self.layers[i-1])
            # first layer
            if i == 1:
                delta_weights[i-1] = np.dot(X.T, dZ) / m
            # hidden layers
            else:
                delta_weights[i-1] = np.dot(self.layers[i-2].T, dZ) / m
            delta_biases[i-1] = np.sum(dZ, axis=0) / m
        
        return delta_weights, delta_biases

    def train(self, X_train, y_train, alpha=0.1, iters=1000):
        for epoch in range(iters):
            # Forward propagation
            self.layers = []
            A = X_train
            for i in range(self.N):
                Z = np.dot(A, self.weights[i]) + self.biases[i]
                if self.activation == 'sigmoid':
                    A = self._sigmoid(Z)
                elif self.activation == 'tanh':
                    A = self._tanh(Z)
                self.layers.append(A)
            y_pred = np.dot(A, self.weights[-1]) + self.biases[-1]
            self.layers.append(y_pred)        
            # Backward propagation
            delta_weights, delta_biases = self._backward(X_train, y_pred, y_train)
                
            # Updating the weights and biases
            for i in range(self.N):
                self.weights[i] -= alpha * delta_weights[i]
                self.biases[i] -= alpha * delta_biases[i]
                  
    def predict(self, X_test):
        A = X_test
        for i in range(self.N):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            if self.activation == 'sigmoid':
                A = self._sigmoid(Z)
            elif self.activation == 'tanh':
                A = self._tanh(Z)
        y_pred = np.dot(A, self.weights[-1]) + self.biases[-1]
        return y_pred
    
    def _loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def _accuracy(self, y_pred, y_true):
        return np.mean((np.round(y_pred) == y_true).astype(int))

nn = NeuralNetwork(n = 2, ne = 4, no = 1, N = 1, activation = 'sigmoid' )
nn.train(X_train, y_train, iters = 1000, alpha = 0.1)
y_pred = nn.predict(X_test)
accuracy = nn._accuracy(y_pred, y_test)
print('Specifically for two layer network; Accuracy with sigmoid with N = 1 :', accuracy)

print('\n----------Using sigmoid function for variable N----------\n')
for i in range(1,6):
    N=i
    nn = NeuralNetwork(n = 2, ne = 4, no = 1, N = N, activation = 'sigmoid' )
    nn.train(X_train, y_train, iters = 1000, alpha = 0.1)
    y_pred = nn.predict(X_test)
    accuracy = nn._accuracy(y_pred, y_test)
    print('\nUsing Sigmoid with N =',N,'\n\n Accuracy:', accuracy)

print('\n----------Using tanh function for variable N----------\n')
for i in range(1,6):
    N=i
    nn = NeuralNetwork(n = 2, ne = 4, no = 1, N = N, activation = 'tanh' )
    nn.train(X_train, y_train, iters = 1000, alpha = 0.1)
    y_pred = nn.predict(X_test)
    accuracy = nn._accuracy(y_pred, y_test)
    print('\nUsing tanh with N =',N,'\n\n Accuracy:', accuracy)

