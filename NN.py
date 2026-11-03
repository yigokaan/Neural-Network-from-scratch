import numpy as np
import json


def sigmoid(z):
    return 1/( 1 + np.e**-z)
    

def Relu(z):
    if z > 0:
       return z
    else:
        return 0

def sigmoid_derivative(z):
    return (1/( 1 + np.e**-z)) * (1-1/( 1 + np.e**-z))

def Relu_derivative(z):
    if z > 0:
       return 1
    else:
        return 0


class NN:
    def __init__(self,inputs):
        self.layers = []
        self.inputs = inputs
        self.learning_rate = 0.1


        
    def add_layer(self,neurons,activation_function,weights = 0,biases = 0):
        if len(self.layers) == 0:
            Added = layer(neurons,activation_function,self.inputs,weights,biases)
            self.layers.append(Added)
        else:
            Added = layer(neurons,activation_function,self.layers[-1].neurons,weights,biases)
            self.layers.append(Added)
        activation_function = activation_function.lower()
        if activation_function == "sigmoid":
            setattr(Added,"activation_func",sigmoid)
            setattr(Added,"activation_derivative",sigmoid_derivative)
        elif activation_function == "relu":
            setattr(Added,"activation_func",Relu)
            setattr(Added,"activation_derivative",Relu_derivative)
        
    
    def Neuron_activator(self,Activated_layer_number,z):
       total = 0
       number = 0
       Activated_layer = self.layers[Activated_layer_number]
       for x in Activated_layer.weights[z]:
          total += x * self.Neural_data[-1][number]
          number += 1
       return total + Activated_layer.biases[z]
    
    def Layer_pass(self,Activated_layer_number):
        Layer_output = []
        number = 0
        Activated_layer = self.layers[Activated_layer_number]
        while number != Activated_layer.neurons:
            Layer_output.append(Activated_layer.activation_func(self.Neuron_activator(Activated_layer_number,number)))
            number += 1
        self.Neural_data.append(Layer_output)

    
    
    def Save(self,name):
        NN_weights = []
        NN_biases = []
        NN_acts = []    
        for Layer in self.layers:
            NN_weights.append(Layer.weights.tolist())
            NN_biases.append(Layer.biases.tolist())
            NN_acts.append(Layer.activation_function)
        dumping_data = {
            "weights": NN_weights,
            "biases": NN_biases,
            "activation_functions": NN_acts
        }
        with open(name+".json", "w", encoding="utf-8") as dosya:
         json.dump(dumping_data, dosya, indent=4)
  


    def Load(self,name):
       with open(name+".json", "r", encoding="utf-8") as f:
         data = json.load(f)
       self.inputs = len(data["weights"][0])
       y = 0
       for x in data["biases"]:
        self.add_layer(len(x),data["activation_functions"][y],weights = np.array(data["weights"][y]),biases=x)
        y += 1 


    def forward_pass(self,input_values):
       self.Neural_data = [input_values]
       number = 0
       while number != len(self.layers):
           self.Layer_pass(number)
           number += 1
       return self.Neural_data[-1]

    #I added back propagation functions but i still am working on it
    def Gradient_descent(self,x):
        Var = np.array(self.layers[x].weights).T @ self.gradients[-1]
        self.gradients.append(self.layers[x].activation_derivative(Var))

    def Back_propagation(self):
       self.Weights_delta = []
       self.gradients = [2 * np.subtract(np.array(self.training_data),np.array(self.Neural_data[-1]))]
       layer_numberp1 = len(self.layers) + 1 
       y = 1
       while y != layer_numberp1:
          self.Gradient_descent(-y)
          y += 1
       y = 1
       while y != layer_numberp1:
          self.weight_change(-y)
          self.bias_change(-y)
          y += 1


    def weight_change(self,xth_layer):
          self.layers[xth_layer].weights += (np.array(self.gradients[xth_layer]) @ np.array(self.Neural_data[xth_layer-1]))*self.learning_rate*(-1)

    
    def bias_change(self,xth_layer):
        self.layers[xth_layer].biases += (np.array(self.gradients[xth_layer]))*self.learning_rate*(-1)


          
          
       
       
        
class layer:
    def __init__(self, neurons, activation_function,Previous,weights,biases):
        self.neurons = neurons
        self.activation_function = activation_function

        if type(weights) is int:
          self.weights = np.random.random((neurons,Previous))
          self.biases = np.random.random(neurons)
        else:
           self.weights = weights
           self.biases = biases
        
    
