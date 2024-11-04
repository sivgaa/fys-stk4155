from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

Niterations = 10000
conv=10**(-4) #konvergenskriterium holder til dette formålet

#eta = 0.5/np.max(EigValues) #learning rate, denne må være mindre enn 2/(den største egenverdien til H) for å kunne konvergere
eta = [1.e-02,5.e-02, 1.e-01, 5.e-01, 1.e+00]
eta = eta / np.max(EigValues) #learning rate < 2/(den største egenverdien til H)

gamma = np.logspace(-5,-1,5) #parameter til GDM

MSE_GD = np.ones([np.size(eta),np.size(gamma)])
R2_GD = np.zeros([np.size(eta),np.size(gamma)])
MSE_OLS = np.ones([np.size(eta),np.size(gamma)])
R2_OLS = np.zeros([np.size(eta),np.size(gamma)])

#delta = 1e-8 #for stabilitet i Adagrad, RMSprop og ADAM
#rho = 0.5 #parameter til RMSprop
#rho1, rho2 = 0.9, 0.99 #parametere til ADAM


for ln in range(np.size(eta)): #løkke over eta-verdier
    for g in range(np.size(gamma)): #løkke over gamma-verdier
        #print('ln,g')
        #print(ln)
        #print(g)
        beta = np.copy(beta_init) #sånn at vi alltid starter med samme utgangpunkt
        beta_prev = np.copy(beta) #starter altså uten momentum hver gang
        i=0 #så vi starter å telle konvergens på nytt
        test = 1 # sånn at løkken starter
        while (i < Niterations and test > conv): #sjekker konvergens
            #Giter = 0.0 #brukes i adagrad
            #first_moment = 0.0 #brukes i ADAM
            #second_moment = 0.0 #brukes i ADAM
            #diff = siv.GD_vanlig(beta,X,y,eta[ln],n)
            diff, beta_prev = siv.GDM(beta,X,y,eta[ln],gamma[g],beta_prev,n)
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
        #print(beta)
        #print(beta_linreg)
        MSE_GD[ln][g] = siv.MSE(ynew,ypredict)
        R2_GD[ln][g] = siv.R2(ynew,ypredict)
        MSE_OLS[ln][g] = siv.MSE(ynew,ypredict2)
        R2_OLS[ln][g] = siv.R2(ynew,ypredict2)
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
#Plotter
#

plt.figure()
#sns.heatmap(data=MSE_GD, annot=True, vmin=18, vmax=18.5)
sns.heatmap(data=MSE_GD, annot=True)
plt.title('MSE GD med momentum')
plt.xlabel('Gamma number')
plt.ylabel('learning rate number')


plt.figure()
sns.heatmap(data=R2_GD, annot=True)
plt.title('R2 GD med momentum')
plt.xlabel('Gamma number')
plt.ylabel('learning rate number')

