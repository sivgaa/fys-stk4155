# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:16:45 2024

@author: siguaa
"""

# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

import siv_funksjoner as siv

STG = False #gjør stochastic gradient descent med minibatcher

#setter opp dataene som vi skal teste mot
n = 100
x = 2*np.random.rand(n,1)
#2. ordens polynom, why not liksom
a=10
b=3
c=1
y = a+b*x+c*x**2+np.random.randn(n,1)

X = np.c_[np.ones((n,1)), x,x*x]
# Hessematrisen kan beregnes direkte, siden den er uavhengig av beta i lineær regresjon
H = (2.0/n)* X.T @ X
# Får tak i egenverdiene og da vil det jo fort vise seg om den er singulær eller ei
EigValues, EigVectors = np.linalg.eig(H)
print(f"Eigenvalues of Hessian Matrix:{EigValues}")

#OLS for å sammenligne med
beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_linreg)

#første gjetning på beta for GD
beta = np.random.randn(3,1)

#eta og gamma er begge navn på "learning raten"
eta = 0.5/np.max(EigValues) #denne må være mindre enn 2/(den støsrte egenverdien til H) for å kunne konvergere
gamma = 0.01 #parameter til GDM
beta_prev = np.copy(beta) #til GDM
delta = 1e-8 #adagrad-parameter (for stabilitet) brukes også i RMSprop
rho = 0.5 #parameter til RMSprop
rho1, rho2 = 0.9, 0.99 #parametere til ADAM

Niterations = 5000
conv=10**(-5)
M = 10 # størrelsen på hver minibatch
m = int(n/M) #antall minibatcher
n_epochs = int(Niterations/m) #antall ganger vi løper gjennom minibatchene 
            #(bare at vi ikke faktisk løper systematisk gjennom, vi trekker tilfeldig så mange ganger som vi har batcher)

t0, t1 = M, n_epochs

#for å ha en learning rate som blir mindre etter hvert som "tiden" går
def learning_schedule(t):
    return t0/(t+t1)

#plot som viser beta-verdienes utvikling gjennom iterasjonene
#fikk koden til plottene av Britt Haanen på kurset 
#prepare plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot([0, Niterations], [a, a], label = 'Nominell verdi')
ax1.plot([0, Niterations], [beta_linreg[0], beta_linreg[0]], label = 'OLS')
ax1.set_title('Beta_0')
ax2.plot([0, Niterations], [b, b], label = 'Nominell verdi')
ax2.plot([0, Niterations], [beta_linreg[1], beta_linreg[1]], label = 'OLS')
ax2.set_title('Beta_1')
ax3.plot([0, Niterations], [c, c], label = 'Nominell verdi')
ax3.plot([0, Niterations], [beta_linreg[2], beta_linreg[2]], label = 'OLS')
ax3.set_title('Beta_2')
fig.suptitle('Gradient Descent (OLS)')


i=0 #for å telle totalt antall iterasjoner
diff_length=1

#
#Stochastic gradient descent
#

if STG:
    #her hadde det vært smart å stokke om på datapunktene våre, så de ikke ligger systematisk. det blir så lokale minibatcher.
    #det skal jeg lære meg en funksjon som gjør...
    while (i < Niterations and diff_length > conv): #sjekker konvergens
        for epoch in range(n_epochs): #løkke over epokene
            Giter = 0.0 #brukes i adagrad
            first_moment = 0.0
            second_moment = 0.0
            for j in range(m): #løkke over minibatchene - ish, vi går ikke gjennom alle
        
                #plot latest value of beta
                ax1.scatter(i,beta[0], label = '_none')
                ax2.scatter(i,beta[1], label = '_none')
                ax3.scatter(i,beta[2], label = '_none') 
               
                random_index = M*np.random.randint(m) #plukker ut tilfeldig element i designmatrisen, dvs velger en tilfeldig batch
                xi = X[random_index:random_index+M] #henter ut tilfeldig element i designmatrisen + påfølgende elementer av en batch-størrelse
                yi = y[random_index:random_index+M] #henter ut tilsvarende y-verdi
                eta = learning_schedule(i)
                #diff = siv.GD_vanlig(beta,xi,yi,eta,n)
                diff, beta_prev = siv.GDM(beta,xi,yi,eta,gamma,beta_prev,n)
                #diff = siv.adaGrad(beta,xi,yi,eta,delta,Giter,n)
                #diff = siv.RMSprop(beta,xi,yi,eta,delta,rho,Giter,n)
                #diff = siv.ADAM(beta,xi,yi,eta,delta,rho1, rho2,first_moment,second_moment, Giter,epoch+1,n)
                beta += diff
                
                i=i+1
        
        #tester konvergens med hele gradienten!
        gradient = (2.0/n)*X.T @ (X @ beta-y)
        diff = -eta*gradient
        diff_length = np.linalg.norm(diff) #brukes for å teste konvergens
else: #helt vanlig gradient descent
    while (i < Niterations and diff_length > conv): #sjekker konvergens
        Giter = 0.0 #brukes i adagrad
        first_moment = 0.0
        second_moment = 0.0
        
        #plot latest value of beta
        ax1.scatter(i,beta[0], label = '_none')
        ax2.scatter(i,beta[1], label = '_none')
        ax3.scatter(i,beta[2], label = '_none') 
        
        eta = learning_schedule(i)
        #diff = siv.GD_vanlig(beta,X,y,eta,n)
        diff, beta_prev = siv.GDM(beta,X,y,eta,gamma,beta_prev,n)
        #diff = siv.adaGrad(beta,X,y,eta,delta,Giter,n)
        #diff = siv.RMSprop(beta,X,y,eta,delta,rho,Giter,n)
        #diff = siv.ADAM(beta,X,y,eta,delta,rho1, rho2,first_moment,second_moment, Giter,epoch+1,n)
        beta += diff
        
        i =i+1
        

ax1.legend() 
ax2.legend()
ax3.legend()  
print(beta)

xnew = np.linspace(0,2,50)
xbnew = np.c_[np.ones((50,1)), xnew, xnew*xnew]
ypredict = xbnew.dot(beta) #GD
ypredict2 = xbnew.dot(beta_linreg) #OLS
plt.figure()
plt.plot(xnew, ypredict, "r-", label='GD')
plt.plot(xnew, ypredict2, "b-",label='OLS') 
plt.plot(x, y ,'ro',label='datapunkter')
#plt.axis([0,2.0,0, 15.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Gradient descent example')
plt.legend()
plt.show()