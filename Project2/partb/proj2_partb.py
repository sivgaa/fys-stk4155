import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from autograd import grad, elementwise_grad
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from copy import deepcopy
import siv_funksjoner_ffnn as sivNN
import siv_funksjoner as siv


def train_network(
    inputs, layers, layers_prev, activation_funcs, target, learning_rate=0.1, gamma=0.1, epochs=1000
):
    
    #
    #Stochastic gradient descent
    #
    m = 10 # andtall minibatcher i SGD
    M = int(np.shape(inputs)[0]/m) # størrelsen på hver minibatch
    
    #Stokker om datasettet for å unngå at minibatchene blir veldig lokale
    #har ingenting å si her ... siden vi har satt opp dataene tilfelig, men kjekt å ha
    inputs, target = shuffle(inputs,target)
    
    layers_prev = np.copy(layers)

    for epoch in range(epochs):
        #ordner med batcher
        random_index = M*np.random.randint(m) #plukker ut tilfeldig element i datasettet, dvs velger en tilfeldig batch
        xi = inputs[random_index:random_index+M] #henter ut tilfeldig element i inputen + påfølgende elementer av en batch-størrelse
        yi = target[random_index:random_index+M] #henter ut tilsvarende y-verdi
        
                      
        #finner gradienter
        layers_grad = sivNN.backpropagation_batch(xi, layers, activation_funcs, yi, activation_ders)
        
        for (W, b), (W_g, b_g), (W_prev, b_prev) in zip(layers, layers_grad, layers_prev):
            tmp = W #trenger å ta vare på denne for å bli den nye "forrige"
            W -= learning_rate*W_g - gamma*(W-W_prev) 
            W_prev = tmp
            tmp = b #trenger å ta vare på denne for å bli den nye "forrige"
            b -= learning_rate*b_g - gamma*(b-b_prev) 
            b_prev = tmp
        
        #For å holde bittelitt øye med at det skjer noe
        if epoch % 100 == 0:
            predictions = sivNN.feed_forward_batch(inputs, layers, activation_funcs)
            mseval = sivNN.mse(predictions, target)
            print(f'MSE =  {mseval:.4f}')

#
#Her begynner vi
#

np.random.seed(42)
#setter opp dataene
n = 100
x = 2*np.random.rand(n,1)
target = siv.poly2gr(x)

#setter opp det nevrale nettverket
network_input_size = np.shape(x)[1] # på en måte antall noder i inputlaget dvs antal features


#antallet lag jeg har øvd med i ukesoppgavene
#var også dette som fungerte best for sigmoid-funksjonen
layer_output_sizes = [4, 1] #må siste lag være 1, siden det jeg sammenligner med er ett tall
activation_funcs = [sivNN.sigmoid,sivNN.linear_act]
activation_ders = [sivNN.sigmoid_der,sivNN.linear_act_der]

"""
# lager litt flere lag da kjører på
layer_output_sizes = [3,3,3,1] #må siste lag være 1, siden det jeg sammenligner med er ett tall
activation_funcs = [sivNN.sigmoid,sivNN.sigmoid,sivNN.sigmoid,sivNN.linear_act]
activation_ders = [sivNN.sigmoid_der,sivNN.sigmoid,sivNN.sigmoid,sivNN.linear_act_der]
"""
"""
# et skikkelig tjukt lag da
layer_output_sizes = [10, 1] #må siste lag være 1, siden det jeg sammenligner med er ett tall
activation_funcs = [sivNN.sigmoid,sivNN.linear_act]
activation_ders = [sivNN.sigmoid_der,sivNN.linear_act_der]
"""
"""
# et skikkelig tynt lag da, 1 virka ikke
layer_output_sizes = [2, 1] #må siste lag være 1, siden det jeg sammenligner med er ett tall
activation_funcs = [sivNN.sigmoid,sivNN.linear_act]
activation_ders = [sivNN.sigmoid_der,sivNN.linear_act_der]
"""
"""
#ett lag med 3 noder var best for sigmoid
layer_output_sizes = [3,1] #må siste lag være 1, siden det jeg sammenligner med er ett tall
activation_funcs = [sivNN.ReLU,sivNN.linear_act]
activation_ders = [sivNN.ReLU_der,sivNN.linear_act_der]
"""

layers, layers_prev = sivNN.create_layers_batch(network_input_size, layer_output_sizes)

#hvor er vi når vi begynner?
predictions = sivNN.feed_forward_batch(x, layers, activation_funcs)
mseval = sivNN.mse(predictions, target)
print(f'MSE ved start =  {mseval:.4f}')

#trener nettverket
train_network(x, layers, layers_prev, activation_funcs, target)


#hvordan går det med et testsett?
#lager et testsett
xnew = np.linspace(0,2,50)
xnew = xnew[:,np.newaxis]
ynew = siv.poly2gr(xnew)

predictions = sivNN.feed_forward_batch(xnew, layers, activation_funcs)
mseval = sivNN.mse(predictions, ynew)
print(f'MSE for testsett  =  {mseval:.4f}')
r2val = siv.R2(ynew,predictions)
print(f'R^2 for testsett  =  {r2val:.4f}')


#plotter det opp, så kanskje det gir litt mening hvorfor det går så dårlig
plt.figure()
plt.plot(xnew, predictions, "r-", label='NN')
plt.plot(x, target ,'ro',label='datapunkter trening')
plt.plot(xnew, ynew ,'bo',label='datapunkter test')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Nevralt nettverk')
plt.legend()
plt.show()





