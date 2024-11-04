import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def linear_act(z):
    return z

def linear_act_der(z):
    return z / z


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return np.exp(-z) / (1 + np.exp(-z))**2

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]

def softmax_der(z):
    sm = softmax(z)
    return sm * (1 - sm)
  
def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def mse(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2*(predict-target)/np.size(target)

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    layers_prev = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size)
        layers.append((W, b))
        layers_prev.append((np.copy(W),np.copy(b)))

        i_size = layer_output_size
    return layers, layers_prev

def feed_forward_batch(input, layers, activation_funcs):
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b #gir en matrise som dar dimensjon (datapunkter, noder i laget)
        a = activation_func(z)
    return a

def feed_forward_saver_batch(input, layers, activation_funcs):
    layer_inputs = [] #a-ene  
    zs = [] #z-ene 
    a = input
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b #gir en matrise som dar dimensjon (datapunkter, noder i laget)
        a = activation_func(z)
        zs.append(z)

    return layer_inputs, zs, a

def cost_batch(layers, input, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)

def backpropagation_batch(
    input, layers, activation_funcs, target, activation_ders, cost_der=mse_der
):
    #finner alle a-er og z-er basert på startgjettene for W og b
    layer_inputs, zs, predict = feed_forward_saver_batch(input, layers, activation_funcs)


    layer_grads = [() for layer in layers] #lager en liste med tupler til å holde dC/dW og dC/db, håper jeg

    
    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i] #henter ut verdiene i det aktuelle laget

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            dC_da = cost_der(predict,target) #i siste laget sammenligner vi direkte med datapunktene våre

        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            (W, b) = layers[i + 1] #henter ut fra laget over
            dC_da = dC_dz @ W.T 
             
        #print('np.shape(dC_da)')
        #print(np.shape(dC_da))
            
        dC_dz = dC_da*activation_der(z) 
        dC_dW = layer_input.T @ dC_dz 
        dC_db = np.sum(dC_dz, axis=0) 
       
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads

