### on ouvre les bibliothèques
import scipy as sp
import pandas as pd
import numpy as np
#import torch
import math
import matplotlib.pyplot as plt
### Import the Gaussian process framework
import GPy
### Import the wavelet transform
import pywt
### And LHS
from pyDOE import lhs
from tqdm import tqdm

### on ouvre notre noyau de Matern
from KernelHaarMatern import KernHaarMatern52

### Definition of the error
def errQ2( estim, ref):
    PRESS = np.sum((estim - ref)**2)
    return(1-PRESS/(ref.shape[0]*np.var(ref)))

def errQ2temp( est , ref):
    output = est[1,:]*0
    for i in range(len(output)):
        output[i] = errQ2(est[:,i],ref[:,i])
    return output

#### Definition of the discontinuous function with linear correlation
def fL(x,t):
    """ Low fidelity fonction """
    return(np.select([t<=x,t>x], [0.5*(6*t-2)**2*np.sin(12*t-4)+10*(t-0.5)-5,3+0.5*(6*t-2)**2*np.sin(12*t-4)+10*(t-0.5)-5]))

def fH(x,t):
    """ High fidelity function """
    return(np.select([t<=x,t>x], [2*fL(x,t)-20*t+20,4+2*fL(x,t)-20*t+20]))


## On génère un sinus avec 1 paramètre pour faire notre simulation de données
# dim = 1
# Nt = 2**4
# t = np.linspace(0, 1,Nt)
# NL = 200
# yH = np.sin( 4*np.pi*(xH/4+1)*t+ xH/10)
# yH = np.sin( 4*np.pi*t )+ (xH/2 -1/4)
# xL = xH
# yL = yH

dim = 1
N = 10
M = 7
NH = 20
Nt = 2**(N+M)
xH = lhs( dim, samples = NH)
tcontinu = np.linspace(0, 1,Nt)    # temps sur échantillonné
t = np.linspace(0, 1, 2**M) # temps sous échantillonné
yHcontinu =  np.sin( 4*np.pi*(xH/4+1)*tcontinu+ xH/10) # fH( xH, tcontinu) # version sur échantillonné de la fonction
yHcontinu = fH( xH, tcontinu)

### On sous échantillonne la fonction
yH = np.zeros((NH,2**M))
for k in range(2**M):
    yH[:,k] = np.sum(yHcontinu[:,k*2**N:(k+1)*2**N],1)* 1/2**N

# # ### Affichage:
# # for k in range(20):
# #     plt.plot(tcontinu, yHcontinu[k,:],'r')
# #     plt.plot(t, yH[k,:],'b')

# # plt.show()
### Comme c'est trop couteux pour le moment
Ndata = 18   # 
Xtest = np.random.uniform(0,1, Ndata).reshape(Ndata,dim)
Exactcontinu = np.sin( 4*np.pi*(Xtest/4+1)*tcontinu+ Xtest/10) # Exactcontinu = fH(Xtest,tcontinu)
Exact = np.zeros((Ndata,2**M))
for k in range(2**M):
    Exact[:,k] = np.sum(Exactcontinu[:,k*2**N:(k+1)*2**N],1)* 1/2**N

#Exact = np.sin( 4*np.pi*(Xtest/4+1)*t+ Xtest/10)
#Exact = np.sin( 4*np.pi*t)+ (Xtest/2 -1/4)
detat = int(t[-1]-t[0])

#Ndata = NH
#Xtest = xH
#Exact = yH
### Décompositon en ondelette
### On ch


### Décompositon en ondelette
### On choisi toujours Haar car c'est celui qui fonctionne pour nos calcules
db1 = pywt.Wavelet('haar') # définition de l'ondelette d'intéret
wlevel = pywt.dwt_max_level(len(t), db1) # nombre de niveau d'ondelettes

## décomposition
### Wavelet transform of the data low and high fidelity
waveletH = pywt.wavedec(yH, db1, mode='constant', level=wlevel)
#waveletL = pywt.wavedec(yL, db1, mode='constant', level=wlevel)

### On calcule les coefficients pour les ondelettes
coeffOndeletteinter = pywt.wavedec(t, db1, mode='constant', level=wlevel)
coeffOndelette = []
for i in range(len(coeffOndeletteinter)):   # iteration sur les niveaux
    lengthelements = len(coeffOndeletteinter[i])
    for k in range(lengthelements):    # iteration sur les positions
        deltatinet = detat/lengthelements 
        coeffOndelette.append((k+1/2)*deltatinet+int(t[0]))

###################################
### Pour la suite on fait de la simple fidélité donc on va s'intéressé que aux données haute fidélité
###################################

### Mise en forme des données pour qu'elles correspondent à un Krigeage simple
sizescalefunction = coeffOndeletteinter[0].shape[0]
NbCO = len(coeffOndelette) - sizescalefunction
NY = NH*NbCO

Ymatrix = np.zeros( ( NH, NbCO))   # initialisation sur les donnée
j = 0   # ititialisation du dernier coefficient modifié
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  waveletH1[i].shape[0]  # On calcule le nombre de données disponibles
    Ymatrix[:,j:(j+tailleData)] = waveletH[i]/np.sqrt(2)  # les coefficents de Haar sont ordonnées dans une matrix
    j = j+tailleData    # le dernier coefficient changé change

Y = Ymatrix.T.reshape(( NY,1))

############### Calculer X
#### Definition des entrées dans notre cas 
X = np.zeros( ( NY, dim+2))

jX = 0
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  waveletH[i].shape[1]  # On calcule le nombre de données disponibles
    X[jX:(jX+NH*tailleData),:dim] = np.repeat(xH, tailleData,axis=1).T.reshape(NH*tailleData,1) # on place les paramètres d'entrée du système
    X[jX:(jX+NH*tailleData),dim] = 2**(wlevel-i)#2**(i-1) # on positionnne les s
    for k in range( tailleData): # on positionne l'ensemble des u
        #X[(jX+k):((jX+(k+1))),2] =  coeffOndelette[j+sizescalefunction+k]*2**(wlevel-i)
        X[(jX+k*NH):((jX+(k+1)*NH)),dim+1] =  k #coeffOndelette[j+sizescalefunction+k]/2**(wlevel-i)#*2**(i-wlevel)
    #j = j+tailleData
    jX = jX+tailleData * NH

# X = np.zeros( ( NY, dim+2)) #initialisation des entrées
# jX = 0
# for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
#     tailleData =  waveletH[i].shape[1]  # On calcule le nombre de données disponibles
#     Ymatrix[:,j:(j+tailleData)] = waveletH[i]   # les coefficents de Haar sont ordonnées dans une matrix
#     X[jX:(jX+NH*tailleData),:dim] = np.repeat(xH, tailleData,axis=1).T.reshape(NH*tailleData,1) # on place les paramètres d'entrée du système
#     X[jX:(jX+NH*tailleData),dim] = 2**(i-1) # on positionnne les s
#     for k in range( tailleData): # on positionne l'ensemble des u
#         X[(jX+k*NH):((jX+(k+1)*NH)),dim+1] = coeffOndelette[j+sizescalefunction+k]*2**(wlevel-i)
#     j = j+tailleData    # le dernier coefficient changé change
#     jX = jX+tailleData * NH   # le dernier coefficient changé change

###########################################################
### Pour i = 0 on passe par la fonction d'echelle donc par un krigeage indépendant
############################################################
Xechell = xH
Yechell = waveletH[0] /np.sqrt(Nt)  ## Normalisation to get the same value for each Nt

active_dimensions = np.arange(0,dim)


kech = GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = False)
mech = GPy.models.GPRegression(X=Xechell, Y=Yechell, kernel=kech)

#m[".*Gaussian_noise"] = m.Y.var()*0.0
#m[".*Gaussian_noise"].fix()

mech.optimize(max_iters = 500, messages=False)  # optimisation des hyperpamètres

#m[".*Gaussian_noise"].unfix()
#m[".*Gaussian_noise"].constrain_positive()

### Pas super utile vu le temps d'optimisation
mech.optimize_restarts(30, optimizer = "bfgs",  max_iters = 1000, verbose=False)#, messages=True)

mu1ech, v1ech = mech.predict(Xtest)

#####################################

#### Definition des entrées dans notre cas 
Xdata = np.zeros( ( Ndata*NbCO, dim+2))

#j = 0   # ititialisation du dernier coefficient modifié
jX = 0
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  coeffOndeletteinter[i].shape[0]  # On calcule le nombre de données disponibles
    Xdata[jX:(jX+Ndata*tailleData),:dim] = np.repeat(Xtest[:Ndata,:], tailleData,axis=1).T.reshape(Ndata*tailleData,1) # on place les paramètres d'entrée du système
    Xdata[jX:(jX+Ndata*tailleData),dim] = 2**(wlevel-i) # on positionnne les s
    for k in range( tailleData): # on positionne l'ensemble des u
        Xdata[(jX+k*Ndata):((jX+(k+1)*Ndata)),dim+1] =  k
    #j = j+tailleData
    jX = jX+tailleData * Ndata

######################################################
##############   Choice of the lerning set ###########
######################################################


# #######    Random state
# setSize = 500   # Definition of the number of element in the learning set
# optimalset = np.random.permutation(NH*NbCO)[:setSize]   # creation of the first set to compare
# kernelHaar = KernHaarMatern52(2+dim,dim, 15)*GPy.kern.Matern52(dim) # definition of the covariance kernel
# #mprior = GPy.models.GPRegression(X=X[optimalset,:], Y=Y[optimalset,:], kernel=kernelHaar)   # Construction of the surrogate model
# ### fit 
# #mprior[".*Gaussian_noise"] = m.Y[optimalset,:].var()*0.01   # definition of the Gaussian noise
# #mprior[".*Gaussian_noise"].fix()    # fit of the Gausian noise
# #mprior.optimize(max_iters = 200,messages=True, optimizer = "bfgs")  # optimization of the hyperparameters
# error = np.sum(mprior.predict(X[optimalset,:], full_cov=False)[1])    # Construction of the error   # np.sum((mprior.predict(X)[0]-Y)**2)

# for iteration in range(10000):
#     nexSet = np.random.permutation(NH*NbCO)[:setSize]   # randomization of the new set
#     mprior = GPy.models.GPRegression(X=X[nexSet,:], Y=Y[nexSet,:], kernel=kernelHaar)   # bulding of the new para
#     ### fit 
#     #mprior[".*Gaussian_noise"].fix()    # fit of the Gausian noise
#     #mprior.optimize(max_iters = 200,messages=False, optimizer = "bfgs")  # optimization of the hyperparameters
#     #mprior[".*variance"].constrain_positive()   # Condition on the variance
#     #mprior[".*lengthscale"].constrain_positive()    # Condition on the lengthscale
#     #mprior.optimize_restarts(3, optimizer = "bfgs",  max_iters = 200, messages=False)
#     newError = np.sum(mprior.predict(X[np.random.permutation(NH*NbCO)[:setSize],:], full_cov=False)[1])   # Set of the new error
#     if newError < error:    # Commpareason with the old best error
#         print(iteration)
#         error = newError
#         optimalset = nexSet

## Pour la fonction d'ondelette
# on définite la fonction de produit d'ondelette:
def psipsiproduct( j, jprime, k, kprime, Nt):
    result = np.zeros(Nt) # initalisation de la sortie
    psi = np.concatenate(( np.zeros(Nt//2**(j+1)) + 1, np.zeros(Nt//2**(j+1)) -1))
    psiprime = np.concatenate(( np.zeros(Nt//2**(jprime+1)) + 1, np.zeros(Nt//2**(jprime+1)) -1))
    size = psi.shape[0]
    sizeprime = psiprime.shape[0]
    psi = np.concatenate((np.zeros(size*k), psi, np.zeros(Nt-size*(k+1))))
    psiprime = np.concatenate((np.zeros(sizeprime*kprime), psiprime, np.zeros(Nt-sizeprime*(kprime+1))))
    return( psi * psiprime/np.sqrt(2**j*2**jprime))

#### Define the computation of the variance
def varianceComp( wlevel, scale, covMat, Ndata, Nt):
    """ This function takes in input the covariance matrix and the sampled elements
        and give in output the variance of the elements.
        Inputs :    wlevel  the number of level of the 
                    scale   the result of the scale function variance
                    covMat  the covariance output of the regression
                    Ndata   the size of data
                    Nt      the length of the time-serie
        Output :    var     the variance in the time space of the elements of X
        """
    var =   np.zeros((Ndata,Nt)) + scale    # Initialisation
    ####    Iteration of the elements of j,k,j',k'
    for j in range(wlevel):
        for jprime in range(wlevel):
            for k in range(2**j):
                for kprime in range(2**jprime):
                    nbvar = (2**j + k -1)*Ndata# on définit la valeur pour le premier élément
                    nbvarprime =  (2**jprime + kprime -1)*Ndata# on définit la valeur pour le deuxième élément
                    variancelocal = np.diag(covMat[nbvar:nbvar+Ndata,0,nbvarprime:nbvarprime+Ndata]).reshape(Ndata,1) # calcule de la covariance local entre  (j,k) et (jprime,kprime) à X constant
                    var = var + variancelocal * psipsiproduct( j, jprime, k, kprime, Nt)    # On ajoute le terme à la somme des termes
    return var

#### Define the computation of the explaine variance
def explainVariance( wlevel, covMat, Ndata, Nt):
    """ This function takes in input the covariance matrix and the sampled elements
        and give in output the variance of the elements.
        Inputs :    wlevel  the number of level of the 
                    covMat  the covariance output of the regression
                    Ndata   the size of data
                    Nt      the length of the time-serie
        Output :    expVar     the variance in the time space of the elements of X

            $ \text{Result} = \sum_{j',k'} Cov(a(j,k),a(j',k')) \psi_{j,k}\psi_{j',k'}
        """
    expVar  =   np.zeros(((Nt-1)*Ndata,1))    # Initialisation
    ####    Iteration of the elements of j,k,j',k'
    for jprime in range(wlevel):    # iteration on jprime
        for kprime in range(2**jprime): #iteration on kprime
            nbvarprime =  (2**jprime + kprime -1)*Ndata# on définit la valeur pour l'element prime
            psiprod = np.zeros(((Nt-1)*Ndata,Nt))
            for j in range( wlevel):    # iteration on j
                for k in range(2**j):   # iteration on k
                    nbvar =  (2**j + k -1)*Ndata    # evaluation of actial state
                    variancelocal = np.diag(covMat[nbvar:nbvar+Ndata,nbvarprime:nbvarprime+Ndata]).reshape(Ndata,1)
                    psiprod[nbvar:(nbvar+Ndata),:] = variancelocal * psipsiproduct( j, jprime, k, kprime, Nt) 
            expVar = expVar + psiprod    # On ajoute le terme à la somme des termes
    return expVar


##### Adaptative way
nbElements = 300   # the number of elements in the learning set
kernelHaar = KernHaarMatern52(2+dim,dim, 2**wlevel)*GPy.kern.Matern52(dim,1,1) # Definition of the kernel
variancediag = np.array([kernelHaar.Kdiag(X[i,:].reshape((1,3))) for i in range(X.shape[0])]).reshape(X.shape[0]) # = np.diag(kernelHaar.K(X))
ratioVector = (1/variancediag * np.abs(Y.reshape(Y.shape[0])))
ratioVector = variancediag* np.abs(Y.reshape(Y.shape[0]))
optimalset = np.argsort(ratioVector, kind = 'mergesort')[-nbElements:]

# for repetition in tqdm(range(10)):   # optimisation loop: we repeat the optimisation in order to improve the learning set
#     corMatrix = kernelHaar.K(X) # definition of the covariance
#     covVector = np.sum(np.abs(corMatrix),1) #explainVariance(wlevel, corMatrix, NH, Nt) # definition of the vector of explaine covariance for each element of the full set
#     ###seuil = np.mean(np.sum(np.abs(corMatrix),1)* (1/X[:,1]))*1.4
#     optimalset = np.argsort(covVector, kind = 'mergesort')[-nbElements:] # elements in the learning set are the most important one in the vector
#     ##optimalser = np.unique(np.heaviside(np.sum(np.abs(corMatrix),1)* (1/X[:,1]) -seuil,1) * np.linspace(0, X.shape[0]-1,X.shape[0]).T)
#     mprior = GPy.models.GPRegression(X=X[optimalset,:], Y=Y[optimalset,:], kernel=kernelHaar)   # bulding of the new para
#     mprior[".*Gaussian_noise"].fix()    # fit of the Gausian noise
#     mprior.optimize()#max_iters = 200,messages=False, optimizer = "bfgs", ipython_notebook=True)  # optimization of the hyperparameters
#     #mprior[".*Gaussian_noise"].unfix()
#     #mprior[".*variance"].constrain_positive()
#     #mprior[".*lengthscale"].constrain_positive()
#     #mprior.optimize_restarts(10, optimizer = "bfgs",  max_iters = 2000, messages=False, ipython_notebook=True)

#### Defition of the new learning set
Xreduce = X[optimalset,:]
Yreduce = Y[optimalset,:]

### On retransforme les donnés pour pouvoir les exploiter dans l'espace considéré.
mu1 = np.zeros(Y.shape)
mu1[optimalset,:] = Yreduce
ydimreduce = [Yechell* np.sqrt(Nt)] ## mean with normalisation of the coefficients
sumsetsize = 0
for i in range(1,wlevel+1):
    setsize = coeffOndeletteinter[i].shape[0]
    inter = np.zeros( ( NH, setsize))
    for j in range(setsize):
        inter[:,j] = mu1[sumsetsize:(sumsetsize+NH)].T
        sumsetsize = sumsetsize + NH
    ydimreduce.append(inter)

yreconstruct =  pywt.waverec(ydimreduce, db1)

couleurs= ['b','r','y','g','purple','orange','navy','magenta','lime','aqua','bisque','royalblue','gray','gold','tan']
for i in range(3):
    plt.plot( t, yH[i,:], color=couleurs[i])
    plt.plot( t, yreconstruct[i,:],'--', color=couleurs[i])
    #plt.fill_between( t, predWmean[i,:] - 1.96*np.sqrt(waveletvar[i,:])/2, predWmean[i,:] + 1.96*np.sqrt(waveletvar[i,:])/2, alpha=0.5, color=couleurs[i])

#plt.savefig("examplesCurves")
plt.show()

### Plot of the chosen elements
plt.plot(ratioVector)
plt.plot(optimalset, ratioVector[optimalset],'o')
plt.show()

######################################################
##############   Regression par GP    ################
######################################################

active_dimensions = np.arange(0,dim)

kernelHaar = KernHaarMatern52(2+dim,dim, Nt)*GPy.kern.Matern52(dim)
m = GPy.models.GPRegression(X=Xreduce, Y=Yreduce, kernel=kernelHaar)

#m[".*Gaussian_noise"] = m.Yreduce.var()*0.0
m[".*Gaussian_noise"].fix()

m.optimize(max_iters = 2000, messages=True, optimizer = "bfgs")  # optimisation des hyperpamètres

m[".*Gaussian_noise"].unfix()
#m[".*Gaussian_noise"].constrain_positive()
m[".*variance"].constrain_positive()
m[".*lengthscale"].constrain_positive()

#
m.optimize_restarts(20, optimizer = "bfgs",  max_iters = 200, messages=True) #,verbose=False)

mu1, v1 = m.predict(Xdata)

### On retransforme les donnés pour pouvoir les exploiter dans l'espace considéré.
waveletPredmu = [mu1ech * np.sqrt(Nt)] ## mean with normalisation of the coefficients
sumsetsize = 0
for i in range(1,wlevel+1):
    setsize = coeffOndeletteinter[i].shape[0]
    inter = np.zeros( ( Ndata, setsize))
    for j in range(setsize):
        inter[:,j] = mu1[sumsetsize:(sumsetsize+Ndata)].T*np.sqrt(2)    # normalisation
        sumsetsize = sumsetsize + Ndata
    waveletPredmu.append(inter)

### calcule de la variance

## pour la fonction d'echelle
vartot = v1ech * db1.wavefun(wlevel)[0]**2
variancetot = np.zeros((Ndata,Nt)) + v1ech * Nt # we take into account the sqrt(Nt) in the normalisation of prediction
    
for j in range(wlevel):
    for jprime in range(wlevel):
        for k in range(2**j):
            for kprime in range(2**jprime):
                nbvar = (2**j + k -1)*Ndata# on définit la valeur pour le premier élément
                nbvarprime =  (2**jprime + kprime -1)*Ndata# on définit la valeur pour le deuxième élément
                variancelocal = np.diag(v1[nbvar:nbvar+Ndata,0,nbvarprime:nbvarprime+Ndata]).reshape(Ndata,1) # calcule de la covariance local entre  (j,k) et (jprime,kprime) à X constant
                variancetot = variancetot + variancelocal * psipsiproduct( j, jprime, k, kprime, Nt)    # On ajoute le terme à la somme des termes
                
waveletvar = abs(variancetot) ### pour que ça soit bien positif


### transformation dans l'espace temporelle
predWmean = pywt.waverec(waveletPredmu, db1)
#### The exact values
waveletExact = pywt.wavedec(Exact, db1, mode='constant', level=wlevel)
### compareason with the prediction in the wavelet space
matrixExact = np.zeros(mu1.shape)
sumsetsize = 0
for i in range(wlevel):
    setsize = coeffOndeletteinter[i].shape[0]
    matrixExact[sumsetsize:(sumsetsize+Ndata*setsize)] = waveletExact[i].reshape(Ndata*setsize,1)
    sumsetsize = sumsetsize + Ndata*setsize
# plt.plot(mu1-matrixExact)
# plt.plot(matrixExact)
# #plt.plot(Y)
# #plt.plot(mu1)
# plt.show()

### Error in the Wavelet space
for i in range(wlevel):
    print(i,np.max(np.abs(waveletExact[i]-waveletPredmu[i])))

for i in range(10):
    plt.plot( t, predWmean[i,:],'--',color=couleurs[i])
    plt.plot( tcontinu, Exactcontinu[i,:],color=couleurs[i])
    plt.plot( t, Exact[i,:], '*', color=couleurs[i])

plt.show()
## plot of the Q2
plt.clf()
plt.plot(t, errQ2temp(predWmean,Exact), label="wavelet")
#plt.plot(t, errQ2temp(PredR.numpy(),Exact.numpy()), label="Mixed")
plt.ylim( 0.70, 1.001)
#plt.legend()
#plt.savefig("Q2NH_7sinus.pdf")
plt.show()
#plt.savefig("WaveletQ2Kriging.pdf")


### Affichage courbes
couleurs= ['b','r','y','g','purple','orange','navy','magenta','lime','aqua','bisque','royalblue','gray','gold','tan']
for i in range(3):
    plt.plot( t, Exact[i,:], color=couleurs[i])
    plt.plot( t, predWmean[i,:],'--', color=couleurs[i])
    #plt.fill_between( t, predWmean[i,:] - 1.96*np.sqrt(waveletvar[i,:])/2, predWmean[i,:] + 1.96*np.sqrt(waveletvar[i,:])/2, alpha=0.5, color=couleurs[i])

#plt.savefig("examplesCurves")
plt.show()
### Affichage de la covariance
plt.imshow(kernelHaar.K(Xdata))
plt.xlabel('X')
plt.ylabel('X')
#plt.savefig("CovarianceMatrix.pdf")
plt.show()

Xsimple = X[:,1:]
toto =KernHaarMatern52(2+1,1, 15).K(X)
plt.imshow(toto)
plt.show()

# #### Save model 
# # save model data
np.save('data_save.npy',xH)
# # save model parameters
np.save('model_save.npy',m.param_array)
# # loading the model
# m2 = GPy.models.GPRegression(X=X, Y=Y, initialize=False, kernel=kernelHaar)
# m2.update_model(False)
# m2.initialize_parameter()
xH2 = np.load('data_save.npy')
tcontinu = np.linspace(0, 1,Nt)    # temps sur échantillonné
t = np.linspace(0, 1, 2**M) # temps sous échantillonné
# m2[:] = np.load('model_save.npy')
# m2.update_model(True)
