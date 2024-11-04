from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

import siv_funksjoner as siv

#seeder koden
np.random.seed(42)


#setter opp dataene
n = 100
x = 2*np.random.rand(n,1)
y = siv.poly2gr(x)

X = np.c_[np.ones((n,1)), x,x*x] #designmatrise
# Hessematrisen kan beregnes direkte, siden den er uavhengig av beta i lineær regresjon
H = (2.0/n)* X.T @ X
# Får tak i egenverdiene og da vil det jo fort vise seg om den er singulær eller ei
EigValues, EigVectors = np.linalg.eig(H)
#print(f"Eigenvalues of Hessian Matrix:{EigValues}")

#lager et testsett
xnew = np.linspace(0,2,50)
xbnew = np.c_[np.ones((50,1)), xnew, xnew*xnew] #designmatrise
ynew = siv.poly2gr(xnew)

#OLS for å sammenligne med
beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_linreg)

#initialiserer
beta_init = np.random.randn(3,1)
i=0 #for å telle totalt antall iterasjoner
test=1 #for å teste konvergens

#
# Setter parametere for kjøringen
#

Niterations = 5000
conv=10**(-4) #konvergenskriterium holder til dette formålet

#eta = 0.5/np.max(EigValues) #learning rate, denne må være mindre enn 2/(den største egenverdien til H) for å kunne konvergere
eta = [1.e-03, 5.e-03, 1.e-02, 5.e-02, 1.e-01, 5.e-01, 1.e+00, 2.e+00]
eta = eta / np.max(EigValues) #learning rate < 2/(den største egenverdien til H)
MSE_GD = np.ones(np.size(eta))
R2_GD = np.zeros(np.size(eta))
MSE_OLS = np.ones(np.size(eta))
R2_OLS = np.zeros(np.size(eta))
#gamma = 0.01 #parameter til GDM
#beta_prev = np.copy(beta) #til GDM
#delta = 1e-8 #for stabilitet i Adagrad, RMSprop og ADAM
#rho = 0.5 #parameter til RMSprop
#rho1, rho2 = 0.9, 0.99 #parametere til ADAM


for ln in range(np.size(eta)): #løkke over eta-verdier
    beta = np.copy(beta_init) #sånn at vi alltid starter med samme utgangpunkt
    i=0 #så vi starter å telle konvergens på nytt
    test = 1 # sånn at løkken starter
    while (i < Niterations and test > conv): #sjekker konvergens
        #Giter = 0.0 #brukes i adagrad
        #first_moment = 0.0 #brukes i ADAM
        #second_moment = 0.0 #brukes i ADAM
        diff = siv.GD_vanlig(beta,X,y,eta[ln],n)
        #diff, beta_prev = siv.GDM(beta,X,y,eta,gamma,beta_prev,n)
        #diff = siv.Adagrad(beta,X,y,eta,delta,Giter,n)
        #diff = siv.RMSprop(beta,X,y,eta,delta,rho,Giter,n)
        #diff = siv.ADAM(beta,X,y,eta,delta,rho1, rho2,first_moment,second_moment, Giter,epoch+1,n)
        beta += diff
        gradient = (2.0/n)*X.T @ (X @ beta-y)
        test = np.linalg.norm(gradient) #brukes for å teste konvergens
        
        i =i+1
    
  

    #tester
    ypredict = xbnew.dot(beta) #GD
    ypredict2 = xbnew.dot(beta_linreg) #OLS
    print(beta)
    print(beta_linreg)
    MSE_GD[ln] = siv.MSE(ynew,ypredict)
    R2_GD[ln] = siv.R2(ynew,ypredict)
    MSE_OLS[ln] = siv.MSE(ynew,ypredict2)
    R2_OLS[ln] = siv.R2(ynew,ypredict2)
    plt.figure()
    plt.plot(xnew, ypredict, "r-", label='GD')
    plt.plot(xnew, ypredict2, "b-",label='OLS') 
    plt.plot(x, y ,'ro',label='datapunkter trening')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'Gradient descent')
    plt.legend()
    plt.show()


#
#Plotter MSE som funksjon av eta
#
plt.figure()
plt.plot(np.log10(eta),MSE_GD,'-',label=f'Gradient descent')
plt.plot(np.log10(eta),MSE_OLS,'-',label=f'OLS')
plt.xlabel(r'log10(learning rate)')
plt.ylabel('MSE')
plt.title('MSE som funksjon av eta for gradient descent')
#plt.ylim([18,18.25])
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(np.log10(eta),R2_GD,'-',label=f'Gradient descent')
plt.plot(np.log10(eta),R2_OLS,'-',label=f'OLS')
plt.xlabel(r'log10(learning rate)')
plt.ylabel('R2')
plt.title('R2 som funksjon av eta for gradient descent')
plt.legend()
plt.tight_layout()
plt.show()