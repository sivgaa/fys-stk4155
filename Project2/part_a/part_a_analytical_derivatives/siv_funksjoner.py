import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import sklearn.linear_model as skl
from sklearn.model_selection import  train_test_split, KFold, cross_val_score
from imageio import imread


#
# Gradient descent-metoder
#

def GD_vanlig(beta, x, y,eta,n):
    gradient = (2.0/n)*x.T @ (x @ beta-y)
    diff = -eta*gradient #vanlig GD
    return diff

def GDM(beta,x,y,eta,gamma,beta_prev,n):
    gradient = (2.0/n)*x.T @ (x @ beta-y)
    diff = -eta*gradient + gamma*(beta - beta_prev)
    beta_prev = np.copy(beta) #tar vare på den forrige verdien
    return diff, beta_prev

def Adagrad(beta,x,y,eta,delta,Giter,n):
    gradient = (2.0/n)*x.T @ (x @ beta-y)
    Giter += gradient*gradient
    diff = -gradient*eta/(np.sqrt(Giter)+delta)
    return diff

def RMSprop(beta,x,y,eta,delta,rho,Giter,n):
    gradient = (2.0/n)*x.T @ (x @ beta-y)
    Giter = (rho*Giter+(1-rho)*gradient*gradient)
    diff = -gradient*eta/(np.sqrt(Giter)+delta)
    return diff

def ADAM(beta,x,y,eta,delta,rho1, rho2,first_moment,second_moment, Giter,i,n):
    gradient = (2.0/n)*x.T @ (x @ beta-y)
    first_moment = rho1*first_moment + (1-rho1)*gradient
    second_moment = rho2*second_moment+(1-rho2)*gradient*gradient
    first_term = first_moment/(1.0-rho1**i)
    second_term = second_moment/(1.0-rho2**i)
    diff = -eta*first_term/(np.sqrt(second_term)+delta)
    return diff

#for å ha en learning rate som blir mindre for hver iterasjon
def learning_schedule(t, t0, t1):
    return t0/(t+t1)
    
#
#Kostfunksjoner og evaluering
#
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
"""
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
"""
def MSE(y_data,y_model):
    #n = np.size(y_model)
    return np.mean((y_data-y_model)**2)



#
# Cross validation
#

def k_fold_cross_val(metode, X, y, lmb,kfold):
    if metode == 'OLS':
        #må gjøre lineær regresjon med scikit-learn
        reg = skl.LinearRegression(fit_intercept=False)
    elif metode =='Ridge':
        reg = skl.Ridge(lmb,fit_intercept=False)
    elif metode == 'Lasso':
        reg = skl.Lasso(lmb,fit_intercept=False, max_iter=20000)
    estimated_mse_folds = cross_val_score(reg, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)
    return estimated_mse_folds

#
#Setter opp data
#

def poly2gr(x):
    a0=10
    a1=3
    a2=1
    y = a0+a1*x+a2*x**2+np.random.randn(np.size(x),1)
    return y

def FrankeFunction(x,y):
    #import numpy as np
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + 0.1*np.random.normal(0,1,x.shape)


def sett_opp_data(datasett,N):
    #setter opp dataene
    if datasett=='franke':
        x = np.sort(np.random.uniform(0, 1, N))     #lager x-verdier som er tilfeldig plassert (uniform fordeling) på [0,1]
        y = np.sort(np.random.uniform(0, 1, N))     #ditto for y-verdiene
        x_mesh, y_mesh = np.meshgrid(x,y)
        z = FrankeFunction(x_mesh, y_mesh)                    #genererer datapunktene fra funksjonen
    elif datasett=='terreng':
        terrain = imread('SRTM_data_Norway_1.tif')
        terrain = terrain[:N,:N]
        # Creates mesh of image pixels
        x = np.linspace(0,1, np.shape(terrain)[0])
        y = np.linspace(0,1, np.shape(terrain)[1])
        x_mesh, y_mesh = np.meshgrid(x,y)
        z = terrain
    return x_mesh, y_mesh, z

#funksjon som setter opp design-matrisen i 2 dimensjoner for polynomer av grad n (tatt fra forelesningsnotatene ke 35)
def create_X(x, y, n ):
    #import numpy as np
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



#
# Diverse brukt i prosjekt 1
#




def sett_opp_beta_koeff(nmax):
    nbetas=0
    for i in range(0,nmax+1): #teller opp hvor mange ledd det blir i det lengste polynomet
        nbetas +=i+1
    beta_koeff_uten_skalering = np.zeros([nbetas,nmax+1]) #setter opp matrisen
    beta_koeff_uten_skalering[:,:]=np.nan                 #fyller matrisen med nan, sånn at det blir smudere å plotte de som ikke skal brukes for alle polynomgrader
    beta_koeff_med_skalering = np.zeros([nbetas,nmax+1]) #setter opp matrisen
    beta_koeff_med_skalering[:,:]=np.nan                 #fyller matrisen med nan, sånn at det blir smudere å plotte de som ikke skal brukes for alle polynomgrader
    return nbetas, beta_koeff_uten_skalering, beta_koeff_med_skalering

#Bruker treningsdataene for å lage modellen
#Gjør prediksjoner og beregner feil for test-dataene
def selve_regresjonen(metode, X_train, y_train, X_test,lmb):
    if metode=='OLS':
        betaOLS = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
        beta=betaOLS
        ytilde = X_train @ beta #prediksjoner for treningsdataene
        ypredict = X_test @ beta #prediksjoner for testdataene
    elif metode=='Ridge':          
        RegRidge = skl.Ridge(lmb,fit_intercept=False)
        RegRidge.fit(X_train,y_train)
        beta=RegRidge.coef_
        ytilde = RegRidge.predict(X_train) #gjør prediksjoner for treningsdataene
        ypredict = RegRidge.predict(X_test) #gjør prediksjoner for testdataene
    elif metode=='Lasso':          
        RegLasso = skl.Lasso(lmb,fit_intercept=False, max_iter=20000)
        RegLasso.fit(X_train,y_train)
        beta=RegLasso.coef_
        ytilde = RegLasso.predict(X_train) #gjør prediksjoner for treningsdataene
        ypredict = RegLasso.predict(X_test) #gjør prediksjoner for testdataene
    return beta, ytilde, ypredict


  

#
# Plottefunksjoner fra prosjekt 1
#     


"""
def lag_plot(x,y,z,tittel):  
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d') 
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(tittel)
    plt.show()
"""    
#
#Plotter betaverdier
#
#tar lambda-verdi som siste argument
def plot_beta_verdier(beta_koeff_uten_skalering,beta_koeff_med_skalering,nbetas,nmax, metode, datasett,kun_skalert,*args):
    plt.figure()
    for i in range(0,nbetas):
        plt.plot(np.arange(0,nmax+1),beta_koeff_uten_skalering[i,:],'b*',label=f'Beta_{i} uten skalering')
        plt.plot(np.arange(0,nmax+1),beta_koeff_med_skalering[i,:],'rx',label=f'Beta_{i} med skalering')
    plt.xlabel(r'Orden av det tilpassede polynomet')
    plt.ylabel(r'Betaverdi')
    if metode == 'OLS':
        plt.title(f'Betaverdier som funksjon av polynomgrad \n for {metode} og {datasett} \n Blå punkter er uten skalering')
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/Betaverdier som funksjon av polynomgrad for '+metode+'_'+datasett+'.png')
    else:
        lmb=args[0]
        print(lmb)
        #nummer som forteller hvilken lambda det er
        #nummer=log(lambda)
        nummer=np.log10(lmb).astype(int).astype(str)
        if kun_skalert:
            plt.title(f'Betaverdier som funksjon av polynomgrad for {metode} og {datasett} \n Lambda = {lmb}')
        else:
            plt.title(f'Betaverdier som funksjon av polynomgrad for {metode} og {datasett} \n Lambda = {lmb} \n Blå punkter er uten skalering')
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/Betaverdier som funksjon av polynomgrad for '+metode+' lambda '+nummer+'_'+datasett+'.png')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   
    plt.show()
    
#
#Plotter bias-variance-tradeoff for bootstrap 
#
def plot_bias_variance(nmax,error, bias, variance,metode,datasett,N):
    plt.figure()
    plt.plot(np.arange(0,nmax+1),error,'*',label=f'MSE')
    plt.plot(np.arange(0,nmax+1),bias,label=f'Bias')
    plt.plot(np.arange(0,nmax+1),variance,label=f'Variance')
    plt.legend()
    plt.xlabel(r'Orden av det tilpassede polynomet')
    plt.title('Bias-variance tradeoff '+metode+' \n'+datasett+ ' med '+ str(N)+' datapunkter' )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'Figures/'+datasett+'/bias-variance-tradeoff som funksjon av polynomgrad for '+metode+'_'+datasett+'.png')
    plt.show()   

#
#Plotter feil-verdier som funksjon av polynomgrad med både skalerte og uskalerte data
#
def plot_feil_polygrad(nlambdas,lambdas,nmax,feil_test_uten_skalering,feil_test_med_skalering,feil,metode,datasett,kun_skalert):
    if kun_skalert:
        plt.figure()
        color = iter(cm.rainbow(np.linspace(0, 1, nlambdas)))
        for l in range(nlambdas):
            if metode != 'OLS': 
                lmb = lambdas[l]
                c = next(color)
                plt.plot(np.arange(0,nmax+1),feil_test_med_skalering[:,l],'-',c=c,label=f'lambda = {lmb}')
            else:
                plt.plot(np.arange(0,nmax+1),feil_test_med_skalering,'-',label='Med skalering')
        plt.xlabel(r'Orden av det tilpassede polynomet')
        plt.ylabel(feil)
        plt.title(feil+' som funksjon av polynomgrad for '+metode+' \n'+datasett+' for skalerte data')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/'+feil+'som funksjon av polynomgrad for '+metode+'_'+datasett+'.png')
        plt.show()
    else:
        plt.figure()
        color = iter(cm.rainbow(np.linspace(0, 1, nlambdas)))
        for l in range(nlambdas):
            if metode != 'OLS': 
                lmb = lambdas[l]
                c = next(color)
                plt.plot(np.arange(0,nmax+1),feil_test_uten_skalering[:,l],'--',c=c,label=f'Uten skalering, lambda = {lmb}')
                plt.plot(np.arange(0,nmax+1),feil_test_med_skalering[:,l],'-',c=c,label=f'Med skalering, lambda = {lmb}')
            else:
                plt.plot(np.arange(0,nmax+1),feil_test_uten_skalering,'--',label='Uten skalering')
                plt.plot(np.arange(0,nmax+1),feil_test_med_skalering,'-',label='Med skalering')
        plt.xlabel(r'Orden av det tilpassede polynomet')
        plt.ylabel(feil)
        plt.title(feil+' som funksjon av polynomgrad for '+metode+' '+datasett)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/'+feil+'som funksjon av polynomgrad for '+metode+'_'+datasett+'.png')
        plt.show()

    
    
#
#Plotter feil som funksjon av labmda med skalerte og uskalerte data
#
def plot_feil_lambda(nlambdas,lambdas,nmax,feil_test_uten_skalering,feil_test_med_skalering,feil,metode,datasett,kun_skalert):
    if kun_skalert:
        plt.figure()
        color = iter(cm.rainbow(np.linspace(0, 1, nmax+1)))
        for n in range(nmax+1):
            c = next(color)
            plt.plot(np.log10(lambdas),feil_test_med_skalering[n,:],'-',c=c,label=f'{n}. ordens polynom')
        plt.xlabel(r'log10(lambda)')
        plt.ylabel(feil)
        plt.title(feil+' \n som funksjon av labmda for '+metode+' \n'+datasett+' for skalerte data')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/'+feil +'som funksjon av lambda for '+metode+'_'+datasett+'.png')
        plt.show()
    else:
        plt.figure()
        color = iter(cm.rainbow(np.linspace(0, 1, nmax+1)))
        for n in range(nmax+1):
            c = next(color)
            plt.plot(np.log10(lambdas),feil_test_uten_skalering[n,:],'--',c=c,label=f'Uten skalering, {n}. ordens polynom')
            plt.plot(np.log10(lambdas),feil_test_med_skalering[n,:],'-',c=c,label=f'Med skalering, {n}. ordens polynom')
        plt.xlabel(r'log10(lambda)')
        plt.ylabel(feil)
        plt.title(feil+' \n som funksjon av labmda for '+metode+' '+datasett)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f'Figures/'+datasett+'/'+feil +'som funksjon av lambda for '+metode+'_'+datasett+'.png')
        plt.show()

            
def plott_data(x,y,z,tittel,datasett):
    print(tittel)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(8))
    ax.set_title(tittel)

    fig.tight_layout()
    plt.savefig(f'Figures/'+datasett+'/'+tittel+'.png')
    plt.show()
    