from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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
beta = np.random.randn(3,1)
i=0 #for å telle totalt antall iterasjoner
test=1 #for å teste konvergens

#
# Setter parametere for kjøringen
#

Niterations = 5000
conv=10**(-4) #konvergenskriterium holder til dette formålet


eta = 0.1 #det var den verdien som fungerte for vandlig GD

#gamma = 0.01 #parameter til GDM
#beta_prev = np.copy(beta) #til GDM
#delta = 1e-8 #for stabilitet i Adagrad, RMSprop og ADAM
#rho = 0.5 #parameter til RMSprop
#rho1, rho2 = 0.9, 0.99 #parametere til ADAM

#plot som viser beta-verdienes utvikling gjennom iterasjonene
#fikk koden til plottene av Britt Haanen på kurset 
#Setter opp plottet
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot([0, Niterations], [10, 10], label = 'Nominell verdi')
ax1.plot([0, Niterations], [beta_linreg[0], beta_linreg[0]], label = 'OLS')
ax1.set_title('Beta_0')
ax1.set_xlabel('Iterasjoner')
ax1.set_ylabel('Verdi')
ax2.plot([0, Niterations], [3, 3], label = 'Nominell verdi')
ax2.plot([0, Niterations], [beta_linreg[1], beta_linreg[1]], label = 'OLS')
ax2.set_title('Beta_1')
ax2.set_xlabel('Iterasjoner')
ax2.set_ylabel('Verdi')
ax3.plot([0, Niterations], [1, 1], label = 'Nominell verdi')
ax3.plot([0, Niterations], [beta_linreg[2], beta_linreg[2]], label = 'OLS')
ax3.set_title('Beta_2')
ax3.set_xlabel('Iterasjoner')
ax3.set_ylabel('Verdi')
fig.suptitle('Stokastisk gradient Descent (OLS)')


#
#Stochastic gradient descent
#

m = 10 # andtall minibatcher i SGD
M = int(n/m) # størrelsen på hver minibatch
n_epochs = int(Niterations/m) #antall ganger vi løper gjennom minibatchene 
            #(bare at vi ikke faktisk løper systematisk gjennom, vi trekker tilfeldig så mange ganger som vi har batcher)

#Stokker om datasettet for å unngå at minibatchene blir veldig lokale
#har ingenting å si her ... siden vi har satt opp dataene tilfelig, men kjekt å ha
x,y,X = shuffle(x,y,X)

     
#for ln in range(np.size(eta)): #løkke over eta-verdier
#    beta = np.copy(beta_init) #sånn at vi alltid starter med samme utgangpunkt
#i=0 #så vi starter å telle konvergens på nytt
#test = 1 # sånn at løkken starter
while (i < Niterations and test > conv): #sjekker konvergens
    #Giter = 0.0 #brukes i adagrad
    #first_moment = 0.0 #brukes i ADAM
    #second_moment = 0.0 #brukes i ADAM
    for j in range(m): #løkke over minibatchene
        #ordner med batcher
        random_index = M*np.random.randint(m) #plukker ut tilfeldig element i designmatrisen, dvs velger en tilfeldig batch
        xi = X[random_index:random_index+M] #henter ut tilfeldig element i designmatrisen + påfølgende elementer av en batch-størrelse
        yi = y[random_index:random_index+M] #henter ut tilsvarende y-verdi
        
        eta = siv.learning_schedule(i, M, n_epochs) 
        
        #plot latest value of beta
        ax1.scatter(i,beta[0], label = '_none')
        ax2.scatter(i,beta[1], label = '_none')
        ax3.scatter(i,beta[2], label = '_none')         
        
        diff = siv.GD_vanlig(beta,xi,yi,eta,M)
        #diff, beta_prev = siv.GDM(beta,X,y,eta,gamma,beta_prev,n)
        #diff = siv.Adagrad(beta,X,y,eta,delta,Giter,n)
        #diff = siv.RMSprop(beta,X,y,eta,delta,rho,Giter,n)
        #diff = siv.ADAM(beta,X,y,eta,delta,rho1, rho2,first_moment,second_moment, Giter,epoch+1,n)
        beta += diff
                    
        i =i+1
 
    #tester konvergens med hele gradienten!
    tst = siv.GD_vanlig(beta,X,y,eta,n)
    test = np.linalg.norm(tst) #brukes for å teste konvergens
 
    
ax1.legend() 
ax2.legend()
ax3.legend() 
print('beta GD') 
print(beta)
    
#tester
ypredict = xbnew.dot(beta) #GD
ypredict2 = xbnew.dot(beta_linreg) #OLS
MSE_GD = siv.MSE(ynew,ypredict)
R2_GD= siv.R2(ynew,ypredict)
MSE_OLS = siv.MSE(ynew,ypredict2)
R2_OLS= siv.R2(ynew,ypredict2)

print('MSE_GD')
print(MSE_GD)
print('MSE_OLS')
print(MSE_OLS)
print('R2_GD')
print(R2_GD)
print('R2_OLS')
print(R2_OLS)

plt.figure()
plt.plot(xnew, ypredict, "r-", label='GD')
plt.plot(xnew, ypredict2, "b-",label='OLS') 
plt.plot(x, y ,'ro',label='datapunkter trening')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Stokastisk gradient descent')
plt.legend()
plt.show()


