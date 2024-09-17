# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Common imports
#import os
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

#funksjon som returnerer z-verdier til Franke-funksjonen (tatt fra forelesningsnotatene til Morten uke 35)
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4 + 0.2*np.random.normal(0,1,x.shape)

#funksjon som setter opp design-matrisen i 2 dimensjoner for polynomer av grad n (tatt fra forelesningsnotatene ke 35)
def create_X(x, y, n ):
    #sjekker om x og y er matriser eller vektorer, konverterer dem til vektorer
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)                  # Antall datapunkter 
	l = int((n+1)*(n+2)/2)		# Antall elementer i beta av orden n i 2 dimensjoner
	X = np.ones((N,l))          # Lager kolonnevektor med samme lengde som x

	for i in range(1,n+1): # i holder orden på graden i leddet
		q = int((i)*(i+1)/2)
		for k in range(i+1): # k passer på fordelingen av potenser mellom x og y
			X[:,q+k] = (x**(i-k))*(y**k)

	return X

#funksjoner for å beregne MSE og R2
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def lag_plot(x,y,z):  #kodesnutt stjålet fra oppgaveteksten til prosjekt 1
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #fant følgende på https://stackoverflow.com/questions/76047803/typeerror-figurebase-gca-got-an-unexpected-keyword-argument-projection
    ax = fig.add_subplot(projection = '3d') 
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


#
#Nu kør vi!
#


nmax = 5                                    #høyeste polynomgrad i modellen
N = 1000                                    #antall datapunkter
metode = 'Ridge'
print(metode)


#setter opp dataene
x = np.sort(np.random.uniform(0, 1, N))     #lager x-verdier som er tilfeldig plassert (uniform fordeling) på [0,1]
y = np.sort(np.random.uniform(0, 1, N))     #ditto for y-verdiene
z = FrankeFunction(x, y)                    #genererer datapunktene fra funksjonen

if metode=='OLS':
    nlambdas=1 #det skal ikke være noen lambda der, men vi må ha en måte å gjøre 1 iterarsjon på
    lambdas=0
elif metode=='Ridge' or 'Lasso':
    nlambdas=10
    lambdas = np.logspace(-4, 5, nlambdas)   


#for å kunne plotte MSE og R2-verdier
MSE_test_uten_skalering = np.zeros([nmax+1,nlambdas])  
MSE_test_med_skalering = np.zeros([nmax+1,nlambdas])   
R2_test_uten_skalering = np.zeros([nmax+1,nlambdas])   
R2_test_med_skalering = np.zeros([nmax+1,nlambdas])   
MSE_train_uten_skalering = np.zeros([nmax+1,nlambdas])  
MSE_train_med_skalering = np.zeros([nmax+1,nlambdas])   
R2_train_uten_skalering = np.zeros([nmax+1,nlambdas])   
R2_train_med_skalering = np.zeros([nmax+1,nlambdas]) 


for l in range(nlambdas): #løkke over lambdaverdier
    if metode != 'OLS': 
        lmb = lambdas[l]
        print(f'lambda = ',lmb)
    #for å kunne plotte betaverdier
    nbetas=0
    for i in range(0,nmax+1): #teller opp hvor mange ledd det blir i det lengste polynomet
        nbetas +=i+1
    beta_koeff_uten_skalering = np.zeros([nbetas,nmax+1]) #setter opp matrisen
    beta_koeff_uten_skalering[:,:]=np.nan                 #fyller matrisen med nan, sånn at det blir smudere å plotte de som ikke skal brukes for alle polynomgrader
    beta_koeff_med_skalering = np.zeros([nbetas,nmax+1]) #setter opp matrisen
    beta_koeff_med_skalering[:,:]=np.nan                 #fyller matrisen med nan, sånn at det blir smudere å plotte de som ikke skal brukes for alle polynomgrader


    for n in range(0,nmax+1): #løkke som løper gjennom polynomgradene
        print(' Polynom av grad ', n)
          
        X = create_X(x, y, n=n)                     #setter designmatrise tilpasset 2D-funksjon
        # splitter dataene i test-data og treningsdata 
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)   #Legg merke til at det er z-verdiene som brukes til y-dataene
        
        #GJØR LINEÆRREGRESJON
        
        #FØRST UTEN SKALERING
    
        if metode=='OLS':
            betaOLS = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
            beta=betaOLS #denne må jo tilpasses hvis vi skal bruke Ridge eller Lasso, da
            ypredict = X_train @ beta #gjør prediksjoner for treningsdataene
        elif metode=='Ridge':          
            RegRidge = skl.Ridge(lmb,fit_intercept=False)
            RegRidge.fit(X_train,y_train)
            beta=RegRidge.coef_
            ypredict = RegRidge.predict(X_train) #gjør prediksjoner for treningsdataene
        elif metode=='Lasso':          
            RegLasso = skl.Lasso(lmb,fit_intercept=False)
            RegLasso.fit(X_train,y_train)
            beta=RegLasso.coef_
            ypredict = RegLasso.predict(X_train) #gjør prediksjoner for treningsdataene
                
        #sparer på beta
        for i in range(0,len(beta+1)):
            beta_koeff_uten_skalering[i,n]=beta[i]
            
            
        #beregner feil for treningsdataene
        
        R2_train_uten_skalering[n,l] = R2(y_train,ypredict) 
        MSE_train_uten_skalering[n,l] = MSE(y_train,ypredict) 
           
        #Gjør prediksjoner og beregner feil for test-dataene
        if metode=='OLS':
            ypredict = X_test @ beta #prediksjoner for testdataene
        elif metode =='Ridge':
            ypredict = RegRidge.predict(X_test) #gjør prediksjoner for testdataene
        elif metode =='Lasso':
            ypredict = RegLasso.predict(X_test) #gjør prediksjoner for testdataene    
        #beregner feil for testdataene
        R2_test_uten_skalering[n,l] = R2(y_test,ypredict) 
        MSE_test_uten_skalering[n,l] = MSE(y_test,ypredict) 
         
         
        
        #HER KOMMER SKALERING
          
        scaler = StandardScaler() #singulær matrise med OLS
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        #print("Feature min values before scaling:\n {}".format(X_train.min(axis=0)))
        #print("Feature max values before scaling:\n {}".format(X_train.max(axis=0)))
        
        #print("Feature min values after scaling:\n {}".format(X_train_scaled.min(axis=0)))
        #print("Feature max values after scaling:\n {}".format(X_train_scaled.max(axis=0)))
     
        if metode=='OLS':
            betaOLS = np.linalg.pinv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train
            beta=betaOLS #denne må jo tilpasses hvis vi skal bruke Ridge eller Lasso, da
            ypredict = X_train_scaled @ beta #gjør prediksjoner for treningsdataene
        elif metode=='Ridge':          
            RegRidge = skl.Ridge(lmb,fit_intercept=False)
            RegRidge.fit(X_train_scaled,y_train)
            beta=RegRidge.coef_
            ypredict = RegRidge.predict(X_train_scaled) #gjør prediksjoner for treningsdataene
        elif metode=='Lasso':          
            RegLasso = skl.Lasso(lmb,fit_intercept=False)
            RegLasso.fit(X_train_scaled,y_train)
            beta=RegLasso.coef_
            ypredict = RegLasso.predict(X_train_scaled) #gjør prediksjoner for treningsdataene  
    
        #sparer på beta
        for i in range(0,len(beta+1)):
            beta_koeff_med_skalering[i,n]=beta[i]
            
        #beregner feil for treningsdataene
        R2_train_med_skalering[n,l] = R2(y_train,ypredict+np.mean(y_train)) 
        MSE_train_med_skalering[n,l] = MSE(y_train,ypredict+np.mean(y_train)) 
           
        #Gjør prediksjoner og beregner feil for test-dataene
        if metode=='OLS':
            ypredict = X_test_scaled @ beta #prediksjoner for testdataene
        elif metode =='Ridge':
            ypredict = RegRidge.predict(X_test_scaled) #gjør prediksjoner for testdataene
        elif metode =='Lasso':
            ypredict = RegLasso.predict(X_test_scaled) #gjør prediksjoner for testdataene
            
        #beregner feil for testdataene
        R2_test_med_skalering[n,l] = R2(y_test,ypredict+np.mean(y_test)) 
        MSE_test_med_skalering[n,l] = MSE(y_test,ypredict+np.mean(y_test))
        
    #
    #Plotter betaverdier
    #
        plt.figure()
    for i in range(0,3):
        plt.plot(np.arange(0,nmax+1),beta_koeff_uten_skalering[i,:],'b--',label=f'Beta_{i} uten skalering')
        plt.plot(np.arange(0,nmax+1),beta_koeff_med_skalering[i,:],'r-',label=f'Beta_{i} med skalering')
    plt.xlabel(r'Orden av det tilpassede polynomet')
    plt.ylabel(r'Betaverdi')
    if metode == 'OLS':
        plt.title(f'Betaverdier som funksjon av polynomgrad for {metode} \n Stiplede linjer er uten skalering')
        plt.savefig(f'Betaverdier som funksjon av polynomgrad for '+metode+'.png')
    else:
        #nummer som forteller hvilken lambda det er
        #nummer=log(lambda)
        nummer=np.log10(lmb).astype(int).astype(str)
        plt.title(f'Betaverdier som funksjon av polynomgrad for {metode} \n Lambda = {lmb} \n Stiplede linjer er uten skalering')
        plt.savefig(f'Betaverdier som funksjon av polynomgrad for '+metode+' lambda '+nummer+'.png')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    plt.show()
      
    """
#Printer verdier for å se litt an
print('R2_test_uten_skalering ', R2_test_uten_skalering)
print('R2_test_med_skalering ', R2_test_med_skalering)
print('MSE_test_uten_skalering ', MSE_test_uten_skalering)
print('MSE_test_med_skalering ', MSE_test_med_skalering)
"""

#
#Lager plot av MSE og R2
#

#
#MSE-verdier som funksjon av polynomgrad
#
plt.figure()
color = iter(cm.rainbow(np.linspace(0, 1, nlambdas)))
for l in range(nlambdas):
    if metode != 'OLS': 
        lmb = lambdas[l]
        c = next(color)
        plt.plot(np.arange(0,nmax+1),MSE_test_uten_skalering[:,l],'--',c=c,label=f'Uten skalering, lambda = {lmb}')
        plt.plot(np.arange(0,nmax+1),MSE_test_med_skalering[:,l],'-',c=c,label=f'Med skalering, lambda = {lmb}')
    else:
        plt.plot(np.arange(0,nmax+1),MSE_test_uten_skalering,'--',label='Uten skalering')
        plt.plot(np.arange(0,nmax+1),MSE_test_med_skalering,'-',label='Med skalering')
plt.xlabel(r'Orden av det tilpassede polynomet')
plt.ylabel(r'MSE')
plt.title(f'MSE som funksjon av polynomgrad for '+metode)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f'MSE som funksjon av polynomgrad for '+metode+'.png')
plt.show()

#
#MSE-verdier som funksjon av labmda
#
if metode !='OLS':
    plt.figure()
    color = iter(cm.rainbow(np.linspace(0, 1, nmax+1)))
    for n in range(nmax+1):
        c = next(color)
        plt.plot(np.log10(lambdas),MSE_test_uten_skalering[n,:],'--',c=c,label=f'Uten skalering, {n}. ordens polynom')
        plt.plot(np.log10(lambdas),MSE_test_med_skalering[n,:],'-',c=c,label=f'Med skalering, {n}. ordens polynom')
 
    plt.xlabel(r'log10(lambda)')
    plt.ylabel(r'MSE')
    plt.title(f'MSE som funksjon av labmda for '+metode)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'MSE som funksjon av lambda for '+metode+'.png')
    plt.show()


#
#R2-verdier som funksjon av polynomgrad
#
plt.figure()
color = iter(cm.rainbow(np.linspace(0, 1, nlambdas)))
for l in range(nlambdas):
    if metode != 'OLS': 
        lmb = lambdas[l]
        c = next(color)
        plt.plot(np.arange(0,nmax+1),R2_test_uten_skalering[:,l],'--',c=c,label=f'Uten skalering, lambda = {lmb}')
        plt.plot(np.arange(0,nmax+1),R2_test_med_skalering[:,l],'-',c=c,label=f'Med skalering, lambda = {lmb}')
    else:
        plt.plot(np.arange(0,nmax+1),R2_test_uten_skalering,'--',label='Uten skalering')
        plt.plot(np.arange(0,nmax+1),R2_test_med_skalering,'-',label='Med skalering')
plt.xlabel(r'Orden av det tilpassede polynomet')
plt.ylabel(r'$R^2$')
plt.title(f'$R^2$ som funksjon av polynomgrad for '+metode)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f'R2 som funksjon av polynomgrad for '+metode+'.png')
plt.show()

#
#R2-verdier som funksjon av labmda
#
if metode !='OLS':
    plt.figure()
    color = iter(cm.rainbow(np.linspace(0, 1, nmax+1)))
    for n in range(nmax+1):
        c = next(color)
        plt.plot(np.log10(lambdas),R2_test_uten_skalering[n,:],'--',c=c,label=f'Uten skalering, {n}. ordens polynom')
        plt.plot(np.log10(lambdas),R2_test_med_skalering[n,:],'-',c=c,label=f'Med skalering, {n}. ordens polynom')
 
    plt.xlabel(r'log10(lambda)')
    plt.ylabel(r'$R^2$')
    plt.title(f'$R^2$ som funksjon av labmda for '+metode)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'R2 som funksjon av lambda for '+metode+'.png')
    plt.show()

    