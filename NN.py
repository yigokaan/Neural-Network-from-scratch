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


        
    def add_layer(self,neurons,activation_function,weights = 0):
        if len(self.layers) == 0:
            Added = layer(neurons,activation_function,self.inputs,weights)
            self.layers.append(Added)
        else:
            Added = layer(neurons,activation_function,self.layers[-1].neurons,weights)
            self.layers.append(Added)
        activation_function = activation_function.lower()
        if activation_function == "sigmoid":
            setattr(Added,"activation_func",sigmoid)
        elif activation_function == "relu":
            setattr(Added,"activation_func",Relu)
        
    
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

    
    #save and load 
    def Save(self,name):
        NN_weights = []
        NN_biases = []
        NN_acts = []    
        for Layer in self.layers:
            NN_weights.append(Layer.weights)
            NN_biases.append(Layer.biases)
            NN_acts.append(Layer.activation_function)
        dumping_data = {
            "weights": NN_weights,
            "biases": NN_biases,
            "activation_functions": NN_acts
        }
        with open(name+".json", "w", encoding="utf-8") as dosya:
         json.dump(dumping_data, dosya, indent=4)
  


    def Load(self,name):
       with open("saf_python_verileri.json", "r", encoding="utf-8") as f:
         data = json.load(f)
       self.inputs = len(data["weights"][0])
       y = 0
       for x in data["biases"]:
        self.add_layer(len(x),data["activation_functions"][y],weights = data["weights"][y])
        y += 1 


    def forward_pass(self,input_values):
       self.Neural_data = [input_values]
       number = 0
       while number != len(self.layers):
           self.Layer_pass(number)
           number += 1
       return self.Neural_data[-1]
       
        

        
class layer:
    def __init__(self, neurons, activation_function,Previous,weights):
        self.neurons = neurons
        self.activation_function = activation_function
        self.biases = np.random.random(neurons)
        if weights == 0:
         self.weights = np.random.random((neurons,Previous))
        else:
           self.weights = weights

