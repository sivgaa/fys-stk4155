# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Common imports
#import os
import numpy as np
#import pandas as pd
#from imageio import imread
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split, KFold#, cross_val_score
from sklearn.preprocessing import StandardScaler#,MinMaxScaler,  Normalizer
from sklearn.utils import resample #til bootstrapen

#Less common imports :P
import siv_funksjoner as siv

#
#Nu kør vi!
#

nmax = 30           #høyeste polynomgrad i modellen
N = 1000            #antall datapunkter
metode = 'OLS'    #OLS, Rige eller Lasso
datasett= 'terreng'  #franke eller terreng
nbootstraps = 0     #NB! Bootstrap fungerer per nå bare for OLS, sett til 0 ellers
kfoldnumber = 0    #kfold cross val antall folds, sett 0 hvis det ikke skal gjøres k-fold cross validation
kun_skalert = True  #kjører med kun skalerte data - stort sett alltid relevant, kunne med fordel snudd denne
plotBeta = False    #genererer plot av betakoeffisienter 
plotModell = True  #plotter dataene og den siste versjonen av den tilpassede modellen - tidkrevende plot. gjøres bare for siste versjon

#setter opp datasettet
x_mesh, y_mesh, z = siv.sett_opp_data(datasett,N)

if metode=='OLS':
    nlambdas=1 #det skal ikke være noen lambda der, men vi må ha en måte å gjøre 1 iterarsjon på
    lambdas=0
elif metode=='Ridge' or 'Lasso':
    nlambdas=10
    lambdas = np.logspace(-6, 3, nlambdas)
    #for bruk ved plotting av favorittmodell
    #nlambdas=1    
    #lambdas = np.logspace(-4, -4, nlambdas)

#for å kunne plotte MSE og R2-verdier
MSE_test_uten_skalering = np.zeros([nmax+1,nlambdas])  
MSE_test_med_skalering = np.zeros([nmax+1,nlambdas])   
R2_test_uten_skalering = np.zeros([nmax+1,nlambdas])   
R2_test_med_skalering = np.zeros([nmax+1,nlambdas])   

if nbootstraps > 0:
    error = np.zeros(nmax+1)
    bias = np.zeros(nmax+1)
    variance = np.zeros(nmax+1)
    
if kfoldnumber > 0:
    kfold = KFold(n_splits = kfoldnumber)

for l in range(nlambdas): #løkke over lambdaverdier
    if metode != 'OLS': 
        lmb = lambdas[l]
        print(f'lambda = ',lmb)
    else: 
        lmb=0
    if plotBeta:
        #for å kunne plotte betaverdier
        nbetas, beta_koeff_uten_skalering, beta_koeff_med_skalering = siv.sett_opp_beta_koeff(nmax)

    for n in range(0,nmax+1): #løkke som løper gjennom polynomgradene
        print(' Polynom av grad ', n)
          
        X = siv.create_X(x_mesh, y_mesh, n=n)  #setter opp designmatrise tilpasset 2D-funksjon
        # splitter dataene i test-data og treningsdata 
        X_train, X_test, y_train, y_test = train_test_split(X,np.ravel(z),test_size=0.2)   #Legg merke til at det er z-verdiene som brukes til y-dataene
        
        #GJØR LINEÆRREGRESJON
        
        #FØRST UTEN SKALERING
        if kun_skalert == False: 
            #her kommer selve regresjonen
            beta, tmp, ypredict = siv.selve_regresjonen(metode, X_train, y_train, X_test,lmb)
            
            #beregner feil for testdataene
            R2_test_uten_skalering[n,l] = siv.R2(y_test,ypredict) 
            MSE_test_uten_skalering[n,l] = siv.MSE(y_test,ypredict)   
            if plotBeta: 
                #sparer på beta
                for i in range(0,len(beta+1)):
                    beta_koeff_uten_skalering[i,n]=beta[i]
        
        #HER KOMMER SKALERING
        scaler = StandardScaler() #NB! gir singulær matrise med OLS
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #skalerer y manuelt
        y_train_scaled = (y_train-np.mean(y_train))/np.std(y_train)
        y_test_scaled = (y_test-np.mean(y_test))/np.std(y_test)
        
        #her kommer selve regresjonen
        #ytilde er prediksjoner for treningsdataene
        #ypredict er prediksjoner for testdataene
        if nbootstraps > 0:
            ypredict = np.empty((y_test.shape[0], nbootstraps))
            for bs in range(nbootstraps):
                X_, y_ = resample(X_train_scaled, y_train_scaled) #stokker om dataene
                beta, ytilde, ypredict[:,bs] = siv.selve_regresjonen(metode, X_, y_, X_test_scaled,lmb) #gjør regresjon og prediksjon
            error[n] = np.mean( np.mean((y_test_scaled[:, np.newaxis] - ypredict)**2, axis=1, keepdims=True) )
            bias[n] = np.mean( (y_test_scaled[:, np.newaxis] - np.mean(ypredict, axis=1, keepdims=True))**2 )
            variance[n] = np.mean( np.var(ypredict, axis=1, keepdims=True) )
            R2_test_med_skalering[n,l] = siv.R2(y_test_scaled,ypredict[:,bs]) #bruker bare siste gangen for at koden skal virke
            MSE_test_med_skalering[n,l] = siv.MSE(y_test_scaled,ypredict[:,bs]) #bruker bare siste gangen for at koden skal virke
        elif kfoldnumber > 0:
            estimated_mse_folds = siv.k_fold_cross_val(metode, X_train_scaled, y_train_scaled,lmb,kfold)
            MSE_test_med_skalering[n,l] = np.mean(-estimated_mse_folds)    
        else: #helt vanlig rett fram etter nesa-versjonen
            beta, ytilde, ypredict = siv.selve_regresjonen(metode, X_train_scaled, y_train_scaled, X_test_scaled,lmb)
            #beregner feil for testdataene
            R2_test_med_skalering[n,l] = siv.R2(y_test_scaled,ypredict) 
            MSE_test_med_skalering[n,l] = siv.MSE(y_test_scaled,ypredict)
        
        if plotBeta:
            #sparer på beta
            for i in range(0,len(beta+1)):
                beta_koeff_med_skalering[i,n]=beta[i]
    #SLUTT PÅ LØKKE OVER POLYNOMGRAD 
    if nbootstraps > 0:
        siv.plot_bias_variance(nmax,error, bias, variance,metode,datasett,N)
    if plotBeta:
        if metode == 'OLS':
            siv.plot_beta_verdier(beta_koeff_uten_skalering,beta_koeff_med_skalering,nbetas,nmax, metode, datasett,kun_skalert)
        else: 
            siv.plot_beta_verdier(beta_koeff_uten_skalering,beta_koeff_med_skalering,nbetas,nmax, metode, datasett,kun_skalert,lmb)

#SLUTT PÅ LØKKE OVER LAMBDA      
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
#Plotter feil-verdier som funksjon av polynomgrad
#
if kfoldnumber > 0 :
    siv.plot_feil_polygrad(nlambdas,lambdas,nmax,MSE_test_uten_skalering,MSE_test_med_skalering,'MSE - k-fold cross validation',metode,datasett,kun_skalert)
elif nbootstraps > 0 :
    siv.plot_feil_polygrad(nlambdas,lambdas,nmax,MSE_test_uten_skalering,MSE_test_med_skalering,'MSE - bootstrap',metode,datasett,kun_skalert)
else:
    siv.plot_feil_polygrad(nlambdas,lambdas,nmax,MSE_test_uten_skalering,MSE_test_med_skalering,'MSE',metode,datasett,kun_skalert)
    siv.plot_feil_polygrad(nlambdas,lambdas,nmax,R2_test_uten_skalering,R2_test_med_skalering,'R2',metode,datasett,kun_skalert)
    
#
#Plotter feil-verdier som funksjon av labmda
#
if metode !='OLS':
    if kfoldnumber > 0:
        siv.plot_feil_lambda(nlambdas,lambdas,nmax,MSE_test_uten_skalering,MSE_test_med_skalering,'MSE - k-fold cross validation',metode,datasett,kun_skalert)
    else:
        siv.plot_feil_lambda(nlambdas,lambdas,nmax,MSE_test_uten_skalering,MSE_test_med_skalering,'MSE',metode,datasett,kun_skalert)
        siv.plot_feil_lambda(nlambdas,lambdas,nmax,R2_test_uten_skalering,R2_test_med_skalering,'R2',metode,datasett,kun_skalert)

#Plotter dataene og modellen
if plotModell:
    #dataene
    #siv.plott_data(X_train[:,1],X_train[:,2],y_train,'data_'+datasett,datasett)
    #den siste modellen jeg laget
    siv.plott_data(X_train[:,1],X_train[:,2],ytilde,'modell_'+datasett,datasett)
    print('MSE - siste modell:')
    print(MSE_test_med_skalering[-1,-1])
