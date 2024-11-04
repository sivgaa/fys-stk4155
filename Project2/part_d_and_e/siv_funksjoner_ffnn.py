import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import autograd.numpy as np 
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns

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

"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return np.exp(-z) / (1 + np.exp(-z))**2
"""
#versjon som skal være mer numerisk stabil
def sigmoid(z):
    s = np.zeros_like(z)
    for i in range(np.shape(z)[0]):
        for j in range(np.shape(z)[1]):
            if z[i][j] < 0:
                s[i][j] = np.exp(z[i][j])/(1+np.exp(z[i][j]))
            else:
                s[i][j] = 1 / (1 + np.exp(-z[i][j]))
    return s
    #return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))
    #return np.exp(-z) / (1 + np.exp(-z))**2

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

#
#Kostfunksjoner og evaluering
#

def mse(predict, target):
    return np.mean((predict - target) ** 2)

def mse_der(predict, target):
    return 2*(predict-target)/np.size(target)
"""
#denne funker med blomster (og muligens bier) når vi har flere mulige utfall
def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)
"""
def accuracy(predictions, targets, threshold=0.5):
    #kunne sikkert valgt andre terskelverdier enn 0.5 også!
    #det blir jo også en slags metaparameter som styrer "sensitiviteten" til modellen
    #det vil påvirke andelen falske negativer osv., selvsagt!!
    pred = predictions > threshold 
    return accuracy_score(pred, targets)


def create_confusion(predictions, targets, threshold = 0.5):
    #kunne sikkert valgt andre terskelverdier enn 0.5 også!
    #det blir jo også en slags metaparameter som styrer "sensitiviteten" til modellen
    #det vil påvirke andelen falske negativer osv., selvsagt!!
    pred = predictions > threshold 
    conf = confusion_matrix(pred, targets)
    conf = conf / conf.sum(axis=1)[:,np.newaxis] #normaliserer
    #conf = conf /conf.sum(axis=1)
    
    plt.figure()
    sns.heatmap(data=conf, annot=True)
    plt.title(f'Normalized confusion matrix \n 0 = malignant, 1 = benign')
    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    #confusion complete


"""
def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cross_entropy_der(predict, target):
    return predict - target
"""

#denne cross entropy funksjonen funker visst bedre og stemmer med scikitlearn
#siden det er np.mean, så fungerer det kanskje uavhengig av dimensjon???
def cross_entropy(predict, target): 
    return -np.mean(target * np.log(predict) + (1 - target) * np.log(1-predict))
"""
#denne gir nan over en lav sko
#alternativt så kan man kanskje legge til en delta i nevnerene?
def cross_entropy_der(predict, target):
    return -np.mean(target/predict-(1-target)/(1-predict))
"""
#litt juks, men håper det kjører nå:
def cross_entropy_der(predict, target):
    der = grad(cross_entropy,0) #deriverer mhp a
    #tmp = der(predict, target)
    #ret = tmp/tmp
    #print(ret)
    #return ret
    #print(der(predict,target))
    return der(predict,target)
    
#til klassifiseringsdata
def cost_batch(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target)

"""
#til numeriske data
def cost_batch(layers, input, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)
"""

#funksjoner for å sette opp nettverket og kjøre

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

#chat gpp mener at layer_inputs = [] lager en liste, og at lister godt kan holde på matriser, så da kanskje det går, da!
def feed_forward_saver_batch(input, layers, activation_funcs):
    layer_inputs = [] #a-ene  #mulig det skal skje noe med dimensjonene her??
    zs = [] #z-ene #mulig det skal skje noe med dimensjonene her??
    a = input
    #print('np.shape(a):')
    #print(np.shape(a))
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = a @ W + b #gir en matrise som dar dimensjon (datapunkter, noder i laget)
        #print('np.shape(z):')
        #print(np.shape(z))
        a = activation_func(z)
        #print('np.shape(a):')
        #print(np.shape(a))

        zs.append(z)

    return layer_inputs, zs, a



