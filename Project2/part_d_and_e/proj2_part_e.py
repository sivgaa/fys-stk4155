import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle, resample
import autograd.numpy as np 
from autograd import grad, elementwise_grad


import siv_funksjoner_ffnn as sivNN
import siv_funksjoner as siv

np.random.seed(42)

def backpropagation_batch(
    input, layers, activation_funcs, target, activation_ders, cost_der=sivNN.cross_entropy_der
):
    #finner alle a-er og z-er basert på startgjettene for W og b
    layer_inputs, zs, predict = sivNN.feed_forward_saver_batch(input, layers, activation_funcs)

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
            
            
        dC_dz = dC_da*activation_der(z) 
        dC_dW = layer_input.T @ dC_dz 
        dC_db = np.sum(dC_dz, axis=0)  
       
        layer_grads[i] = (dC_dW, dC_db)
    return layer_grads

def train_network(
    inputs, layers, layers_prev, activation_funcs, target, threshold, learning_rate=0.1, gamma = 0.0001, epochs=10000
):
    #learning_rate=0.001 original
    #epochs=100 orignial
       
    #
    #Stochastic gradient descent
    #
    m = 20 # antall minibatcher i SGD
    M = int(np.shape(inputs)[0]/m) # størrelsen på hver minibatch
    
    #Stokker om datasettet for å unngå at minibatchene blir veldig lokale
    #har ingenting å si her ... siden vi har satt opp dataene tilfelig, men kjekt å ha
    inputs, target = shuffle(inputs,target)
        
    for epoch in range(epochs):
        #ordner med batcher
        random_index = M*np.random.randint(m) #plukker ut tilfeldig element i datasettet, dvs velger en tilfeldig batch
        xi = inputs[random_index:random_index+M] #henter ut tilfeldig element i inputen + påfølgende elementer av en batch-størrelse
        yi = target[random_index:random_index+M] #henter ut tilsvarende y-verdi
        
        #layers_grad2 = cheating(layers, activation_funcs, inputs, target)
        layers_grad = backpropagation_batch(xi, layers, activation_funcs, yi, activation_ders)
        #print(layers_grad2[0][0] - layers_grad[0][0])
        for (W, b), (W_g, b_g), (W_prev, b_prev) in zip(layers, layers_grad, layers_prev):
            #print(W_g)
            """
            W -= learning_rate*W_g #foreløpig uten momentum her, altså
            b -= learning_rate*b_g #foreløpig uten momentum her, altså
            """
            tmp = np.copy(W) #trenger å ta vare på denne for å bli den nye "forrige"
            W -= learning_rate*W_g - gamma*(W-W_prev) 
            W_prev = np.copy(tmp)
            tmp = np.copy(b) #trenger å ta vare på denne for å bli den nye "forrige"
            b -= learning_rate*b_g - gamma*(b-b_prev) 
            b_prev = np.copy(tmp)            
        
        #For å holde bittelitt øye med at det skjer noe
        if epoch % 100 == 0:    
            predictions = sivNN.feed_forward_batch(inputs, layers, activation_funcs)
            #print(predictions[1])
            #print(target[1])
            acc = sivNN.accuracy(predictions, target, threshold)
            print(f'Accuracy =  {acc:.4f}')
        
#
#Settter opp data
#

#kode for import av data fra mortens notater uke 42 
#Der finnes det eksempel med logistisk regresjon 
wisconsin = load_breast_cancer()
x = wisconsin.data
target = wisconsin.target
target = target.reshape(target.shape[0], 1)
#print(np.shape(x))

#SKALERER DATAENE
#MinMaxScaler ble importert i eksemelkoden til Morten, antar den er fornufig her
#siden target er 0  og 1, så trenger vi ikke skalere det, går jeg ut ifra 
scaler = MinMaxScaler() 
scaler.fit(x)
x_scaled = scaler.transform(x)

#Deler i test og treningssett
x_train, x_test, target_train, target_test = train_test_split(x_scaled, target, test_size=0.2)

network_input_size = np.shape(x)[1] # på en måte antall noder i inputlaget dvs antal features

#I kreftdatene er 0=positiv(malign/kreft) og 1 = negativ(benign/ikke kreft)
#Det betyr at en høy terskel gir flere positive prediksjoner og falske positive
#Det burde også gjøre ta det blir færre falske negative
threshold = 0.5 #terskelverdien for å gjøre prediksjonene binære

#
#En logistisk regresjon er bare et nevralt nettverk ut en skjulte lag
#med sigmoidfunksjonen i det siste(eneste) laget
#

#ingen skjultelag
layer_output_sizes = [1] #den siste må være 1 og siste activation skal være sigmoid
activation_funcs = [sivNN.sigmoid]
activation_ders = [sivNN.sigmoid_der]



layers, layers_prev = sivNN.create_layers_batch(network_input_size, layer_output_sizes)
#hvor er vi når vi begynner?
predictions = sivNN.feed_forward_batch(x_train, layers, activation_funcs)
print(f'Accuracy first guess =  {sivNN.accuracy(predictions, target_train, threshold):.4f}')
#print(sivNN.accuracy(predictions, target_train))


#trener nettverket
train_network(x_train, layers, layers_prev, activation_funcs, target_train, threshold, learning_rate = 1.0, gamma = 0.00001)


#evaluerer med testsett
#accuracy:
predictions = sivNN.feed_forward_batch(x_test, layers, activation_funcs)
print(f'Accuracy test data =  {sivNN.accuracy(predictions, target_test, threshold):.4f}')

#confusion matrix
sivNN.create_confusion(predictions, target_test, threshold)





