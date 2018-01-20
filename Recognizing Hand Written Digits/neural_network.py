import load_data
import numpy as np
import random

training_data, validation_data, test_data = load_data.organize_data()

def sigmoid (x):
    return (1./(1+np.exp(-x)))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class neural_network(object) :
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights= [np.random.randn(x,y) for (x,y) in zip(sizes[1:],sizes[:-1])]
    
    def gradient_descent(self, training_data, epochs, mini_batch_size, alpha, test_data = test_data) :
        n = sum(1 for _ in training_data)
        n_test = sum(1 for _ in test_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[x : x+mini_batch_size] for x in range(0,n,mini_batch_size) ]
            for mini_batch in mini_batches :
                self.traverse(mini_batch,alpha)
            if test_data :
                print("Epoch {} : {}/{}".format(i,self.evaluate(test_data),n_test))
            else: print("Epoch {} complete".format(i))
    
    def traverse (self, mini_batch,alpha) :
        error_b = [np.zeros(bias.shape) for bias in self.biases]
        error_w = [np.zeros(weight.shape) for weight in self.weights]
        for (x,y) in mini_batch :
            delta_w, delta_b = self.backpropagate(x,y)
            error_w = [e + d for e,d in zip(error_w,delta_w)]
            error_b = [e + d for e,d in zip(error_b,delta_b)]
        self.weights = [(weight - ((1./len(mini_batch)) * alpha )*error) for weight, error in zip(self.weights,error_w)]
        self.biases = [(bias - ((1./len(mini_batch)) * alpha )*error) for bias, error in zip(self.biases,error_b)]
    
    def backpropagate (self, x, y):
        delta_b = [np.zeros(bias.shape) for bias in self.biases]
        delta_w = [np.zeros(weight.shape) for weight in self.weights]
        #z = x
        zs=[]
        activation = x
        activations = [activation]
        for w, b in zip(self.weights,self.biases):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (activations[-1]-y) * sigmoid_prime(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta,activations[-2].T)
        for l in range(2,self.num_layers):
            z=zs[-l]
            delta = np.dot(self.weights[-l+1].T,delta)*sigmoid_prime(z)
            delta_b[-l]= delta
            delta_w[-l]= np.dot(delta,activations[-l-1].T)
        return (delta_w, delta_b)
    
    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def evaluate(self,test_data):
        test_result = [(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        return sum(int(x==y) for x,y in test_result)

network = neural_network([784,40,40,10])

network.gradient_descent(training_data, 30, 10, 2.5, test_data=test_data)