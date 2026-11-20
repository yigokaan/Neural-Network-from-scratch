import numpy as np
import json
import mnist
import time
print("//starting//")


def PreSoftmax(self,x):
    self.Softmax_activated = True
    return x
   
def Softmax(array):
    max = np.max(array)
    total = 0
    for x in array:
        total += np.exp(x - max)
    for x in range(len(array)):
        array[x] = np.exp(array[x] - max) / total
    return array


#temporary
def Softmax_derivative(p):
    return np.diag(p) - np.outer(p, p)


def sigmoid(z):
    return 1/( 1 + np.e**-z)
    

def Relu(z):
    return (z * (z > 0))

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    return s * (1 - s)

def Relu_derivative(z):
    return (z > 0)


def prediction_accuracy(output,labels):
    if np.argmax(output) == np.argmax(labels):
        return 1
    else:
        return 0

class NN:
    def __init__(self,inputs):
        self.layers = []
        self.inputs = inputs
        self.learning_rate = 0.00005
        self.Softmax_activated = False


        
    def add_layer(self,neurons,activation_function,weights = 0,biases = 0,initialization = "xavier",uniform=False):
        activation_function = activation_function.lower()
        if len(self.layers) == 0:
            Added = layer(neurons,activation_function,self.inputs,weights,biases,initialization,uniform)
            self.layers.append(Added)
        else:
            Added = layer(neurons,activation_function,self.layers[-1].neurons,weights,biases,initialization,uniform)
            self.layers.append(Added)
        if activation_function == "sigmoid":
            setattr(Added,"activation_func",sigmoid)
            setattr(Added,"activation_derivative",sigmoid_derivative)
        elif activation_function == "relu":
            setattr(Added,"activation_func",Relu)
            setattr(Added,"activation_derivative",Relu_derivative)
        elif activation_function == "softmax":        
            setattr(Added,"activation_func",lambda x :PreSoftmax(self,x))
            setattr(Added,"activation_derivative",Softmax_derivative)


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
        pre_out = []
        number = 0
        Activated_layer = self.layers[Activated_layer_number]
        while number != Activated_layer.neurons:
            pre_act = self.Neuron_activator(Activated_layer_number,number)
            pre_out.append(pre_act)
            Layer_output.append(Activated_layer.activation_func(pre_act))
            number += 1
        if self.Softmax_activated:
            Layer_output = Softmax(Layer_output)
            self.Softmax_activated = False
        self.PreNeural_data.append(pre_out)
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
       self.PreNeural_data = [input_values]
       number = 0
       while number != len(self.layers):
           self.Layer_pass(number)
           number += 1
       return self.Neural_data[-1]

    def Gradient_descent(self,x):
        Var = np.array(self.layers[x+1].weights).T @ self.gradients[-1]
        for i in range(len(Var)):
           Var[i] *= self.layers[x].activation_derivative(self.PreNeural_data[x][i])
        self.gradients.append(Var)


    def Back_propagation(self,output_number):
       self.Weights_delta = []
       num = 0
       #Planning to turn the loss function derivative inside to self.loss() to make it modifiable
       k = 2 * (
          np.array(self.Neural_data[-1])
          - np.array(self.training_data[output_number]))
       """
       if not self.layers[-1].activation_function == "softmax":
        for x in k:
           k[num] = x * self.layers[-1].activation_derivative(self.PreNeural_data[-1][num])
           num += 1
       else:
           k = self.layers[-1].activation_derivative(
            self.PreNeural_data[-1]) @ k"""
       self.gradients = [k]
       layer_numberp1 = len(self.layers)
       y = 1
       while y != layer_numberp1:
          self.Gradient_descent(-y-1)
          y += 1
       self.gradients.reverse()
       y = 1
       while y != layer_numberp1+1:
          self.weight_change(-y)
          self.bias_change(-y)
          y += 1

       self.total_prediction_accuracy += prediction_accuracy(self.Neural_data[-1],self.training_data[output_number])
       self.tries += 1
       if self.tries == 100:
           print(self.total_prediction_accuracy)
           self.tries = 0
           self.total_prediction_accuracy = 0
           print(self.average_loss/100)
           self.average_loss = 0
       self.average_loss += sum(k**2)
       #print("Loss:", sum(k**2))
       #print("OUTPUT:", self.Neural_data[-1])
       #print("TARGET:", self.training_data[output_number])
       #time.sleep(1)


    def weight_change(self,xth_layer):
          self.layers[xth_layer].weights += np.outer(np.array(self.gradients[xth_layer]),np.array(self.Neural_data[xth_layer-1]))*self.learning_rate*(-1)
          #print(np.outer(np.array(self.gradients[xth_layer]),np.array(self.Neural_data[xth_layer-1]))*self.learning_rate*(-1))
          #print(self.Neural_data[xth_layer-1])
          #time.sleep(1)

    def bias_change(self,xth_layer):
        self.layers[xth_layer].biases += (np.array(self.gradients[xth_layer]))*self.learning_rate*(-1)


    def Train(self,input_data,output_data):
        self.training_data = output_data
        k = 0
        self.total_prediction_accuracy,self.tries,self.average_loss = 0,0,0
  
        for x in input_data:
            self.forward_pass(np.array(x).flatten())
            self.Back_propagation(k)
            k += 1

    def Output(self):
        print("Underwork-this class is for ")

           
       
       
        
class layer:
    def __init__(self, neurons, activation_function,Previous,weights,biases,initialization,uniform):
        self.neurons = neurons
        self.activation_function = activation_function
       
        if type(weights) is int: 
          if initialization == "xavier":
            if uniform == False:
                    std = np.sqrt(2 / neurons + Previous)
                    self.weights = np.random.normal(0,std,(neurons,Previous))
            else:
                    std = np.sqrt(6 / neurons + Previous)
                    self.weights = np.random.uniform(-std,std,(neurons,Previous))
              
          elif initialization == "he":
            if uniform == False:
                    std = np.sqrt(2 / Previous)
                    self.weights = np.random.normal(0,std,(neurons,Previous))
            else:
                    std = np.sqrt(6 / Previous)
                    self.weights = np.random.uniform(-std,std,(neurons,Previous))
                      
          self.biases = np.zeros(neurons)
        else:
           self.weights = weights
           self.biases = biases
        
