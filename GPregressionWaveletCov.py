### on ouvre les bibliothèques
import scipy as sp
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
### Import the Gaussian process framework
import GPy
### Import the wavelet transform
import pywt
### And LHS
from pyDOE import lhs

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


## On génère un sinus avec 1 paramètre pour faire notre simulation de données
dim = 1
Nt = 2**6
t = np.linspace(0, 1,Nt)
NH = 5
NL = 23
xH = lhs( dim, samples = NH)
yH = np.sin( 4*np.pi*xH*t+ xH/2)
#yH = np.sin( 4*np.pi*t)+ (xH/2 -1/4)
xL = xH
yL = yH
Ndata = 10

### Comme c'est trop couteux pour le moment
Ndata = 100
Xtest = np.random.uniform(0,1, Ndata).reshape(Ndata,dim)
Exact = np.sin( 4*np.pi*Xtest*t+ Xtest/2)
#Exact = np.sin( 4*np.pi*t)+ (Xtest/2 -1/4)
detat = int(t[-1]-t[0])


### Décompositon en ondelette
### On ch

### Décompositon en ondelette
### On choisi toujours Haar car c'est celui qui fonctionne pour nos calcules
db1 = pywt.Wavelet('haar') # définition de l'ondelette d'intéret
wlevel = pywt.dwt_max_level(len(t), db1) # nombre de niveau d'ondelettes

## décomposition
### Wavelet transform of the data low and high fidelity
waveletH = pywt.wavedec(yH, db1, mode='constant', level=wlevel)
waveletL = pywt.wavedec(yL, db1, mode='constant', level=wlevel)

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
sizeechel = len(coeffOndeletteinter[0])     # taille des coefficients d'echelle
NY = NH*NbCO
Ymatrix = np.zeros( ( NH, NbCO))   # initialisation sur les donnée
X = np.zeros( ( NY, dim+2)) #initialisation des entrées
j = 0   # ititialisation du dernier coefficient modifié
jX = 0
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  waveletH[i].shape[1]  # On calcule le nombre de données disponibles
    Ymatrix[:,j:(j+tailleData)] = waveletH[i]   # les coefficents de Haar sont ordonnées dans une matrix
    X[jX:(jX+NH*tailleData),:dim] = np.repeat(xH, tailleData,axis=1).T.reshape(NH*tailleData,1) # on place les paramètres d'entrée du système
    X[jX:(jX+NH*tailleData),dim] = 2**(i-1) # on positionnne les s
    for k in range( tailleData): # on positionne l'ensemble des u
        X[(jX+k*NH):((jX+(k+1)*NH)),dim+1] = coeffOndelette[j+sizescalefunction+k]
    j = j+tailleData    # le dernier coefficient changé change
    jX = jX+tailleData * NH   # le dernier coefficient changé change

#X[:,2] = X[:,1]*X[:,2]-1/2 # pour le cas où les indices sont des entier
#### Pour avoir les sorties sous forme de vecteurs
Y = Ymatrix.T.reshape(( NY,1))

###########################################################
### Pour i = 0 on passe par la fonction d'echelle donc par un krigeage indépendant
############################################################
Xechell = xH
Yechell = waveletH[0]

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

j = 0   # ititialisation du dernier coefficient modifié
jX = 0
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  coeffOndeletteinter[i].shape[0]  # On calcule le nombre de données disponibles
    Xdata[jX:(jX+Ndata*tailleData),:dim] = np.repeat(Xtest[:Ndata,:], tailleData,axis=1).T.reshape(Ndata*tailleData,1) # on place les paramètres d'entrée du système
    Xdata[jX:(jX+Ndata*tailleData),dim] = 2**(i-1) # on positionnne les s
    for k in range( tailleData): # on positionne l'ensemble des u
        Xdata[(jX+k*Ndata):((jX+(k+1)*Ndata)),dim+1] = coeffOndelette[j+sizescalefunction+k]
    j = j+tailleData
    jX = jX+tailleData * Ndata


######################################################
######### Regression par GP ##########################
######################################################

active_dimensions = np.arange(0,dim)

kernelHaar = KernHaarMatern52(2+dim,dim, 15)*GPy.kern.Matern52(dim)
m = GPy.models.GPRegression(X=X, Y=Y, kernel=kernelHaar)

m[".*Gaussian_noise"] = m.Y.var()*0.0
m[".*Gaussian_noise"].fix()

m.optimize(max_iters = 2000,messages=True, optimizer = "scg")  # optimisation des hyperpamètres

#m[".*Gaussian_noise"].unfix()
#m[".*Gaussian_noise"].constrain_positive()
#m[".*variance"].constrain_positive()

#
# m.optimize_restarts(20, optimizer = "scg",  max_iters = 2000, messages=True) #,verbose=False)

mu1, v1 = m.predict(Xdata)

### On retransforme les donnés pour pouvoir les exploiter dans l'espace considéré.
waveletPredmu = [mu1ech] ## moyenne 
sumsetsize = 0
for i in range(1,wlevel+1):
    setsize = coeffOndeletteinter[i].shape[0]
    inter = np.zeros( ( Ndata, setsize))
    for j in range(setsize):
        inter[:,j] = mu1[sumsetsize:(sumsetsize+Ndata)].T
        sumsetsize = sumsetsize + Ndata
    waveletPredmu.append(inter)

### calcule de la variance

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

## pour la fonction d'echelle
vartot = v1ech * db1.wavefun(wlevel)[0]**2
variancetot = np.zeros((Ndata,Nt)) + v1ech
    
for j in range(wlevel):
    for jprime in range(wlevel):
        for k in range(2**j):
            for kprime in range(2**jprime):
                nbvar = (2**j + k -1)*Ndata# on définit la valeur pour le premier élément
                nbvarprime =  (2**jprime + kprime -1)*Ndata# on définit la valeur pour le deuxième élément
                variancelocal = np.diag(v1[nbvar:nbvar+Ndata,0,nbvarprime:nbvarprime+Ndata]).reshape(Ndata,1) # calcule de la variance local pour j,k,jprime,kprime
                variancetot = variancetot + variancelocal * psipsiproduct( j, jprime, k, kprime, Nt)
                
waveletvar = abs(variancetot) ### pour que ça soit bien positif


### transformation dans l'espace temporelle
predWmean = pywt.waverec(waveletPredmu, db1)
#### The exact values
waveletExact = pywt.wavedec(Exact, db1, mode='constant', level=wlevel)

### erreur au niveau du Q2:
for i in range(len(waveletExact)):
    print(errQ2temp(waveletPredmu[i], waveletH[i]))


## plot of the Q2
plt.clf()
plt.plot(t, errQ2temp(predWmean,Exact), label="wavelet")
#plt.plot(t, errQ2temp(PredR.numpy(),Exact.numpy()), label="Mixed")
plt.ylim( 0.90, 1.001)
plt.legend()
#plt.savefig("Q2NH_7sinus.pdf")
plt.show()
#plt.savefig("WaveletQ2Kriging.pdf")


### Affichage courbes
couleurs= ['b','r','y','g','purple','orange','navy','magenta','lime','aqua','bisque','royalblue','gray','gold','tan']
for i in range(10):
    plt.plot( t, Exact[i,:], color=couleurs[i])
    plt.plot( t, predWmean[i,:],'--', color=couleurs[i])
    #plt.fill_between( t, predWmean[i,:] - 1.96*np.sqrt(waveletvar[i,:])/2, predWmean[i,:] + 1.96*np.sqrt(waveletvar[i,:])/2, alpha=0.5, color=couleurs[i])

#plt.savefig("examplesCurves")
plt.show()

### Affichage de la covariance
plt.imshow(kernelHaar.K(X))
plt.xlabel('X')
plt.ylabel('X')
#plt.savefig("CovarianceMatrix.pdf")
plt.show()

#### Save model from 
np.save('model_save.npy',m.param_array)
# loading the model
m = GPy.models.GPRegression(X=X, Y=Y, initialize=False, kernel=kernelHaar)
m.update_model(False)
m.initialize_parameter()
m[:] = np.load('model_save.npy')
m.update_model(True)
