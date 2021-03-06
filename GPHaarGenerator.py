import numpy as np
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
from numba import jit

#### Definition of the discontinuous function with linear correlation
def fL(x,t):
    """ Low fidelity fonction """
    return(np.select([t<=x,t>x], [0.5*(6*t-2)**2*np.sin(12*t-4)+10*(t-0.5)-5,3+0.5*(6*t-2)**2*np.sin(12*t-4)+10*(t-0.5)-5]))

def fH(x,t):
    """ High fidelity function """
    return(np.select([t<=x,t>x], [2*fL(x,t)-20*t+20,4+2*fL(x,t)-20*t+20]))


## On génère un sinus avec 1 paramètre pour faire notre simulation de données
dim = 1
N = 4
M = 4
Nt = 2**(N+M)
tcontinu = np.linspace(0, 1,Nt)    # temps sur échantillonné
t = np.linspace(0, 1, 2**M) # temps sous échantillonné
yHcontinu = fH( 0.5, tcontinu) # version sur échantillonné de la fonction

### On sous échantillonne la fonction
yH = np.zeros((2**M))
for k in range(2**M):
    yH[k] = np.sum(yHcontinu[k*2**N:(k+1)*2**N])* 1/2**N

####### Compareason ##############
# plt.plot(tcontinu, yHcontinu,'r')
# plt.plot(t,yH,'b')
# plt.show()


####### Conversion into Wavelet coefficients #####
db1 = pywt.Wavelet('haar') # définition de l'ondelette d'intéret
wlevel = pywt.dwt_max_level(len(t), db1) # nombre de niveau d'ondelettes

## décomposition
### Wavelet transform of the data low and high fidelity
waveletH1 = pywt.wavedec(yH, db1, mode='constant', level=wlevel)

# waveletH2 = pywt.wavedec(yH, db1, mode='constant', level=wlevel)

# ### Compareason between the 2 wavelet transform
# Lerror = []
# for i in range(wlevel):
#     Lerror.append(np.sum((waveletH1[i]-waveletH2[i])**2))

### Construction of the input and outputs coefficients

### Computation of the wavelet coefficients
coeffOndeletteinter = pywt.wavedec(t, db1, mode='constant', level=wlevel)
coeffOndelette = []
detat = int(t[-1]-t[0])
for i in range(len(coeffOndeletteinter)):   # iteration over the level
    lengthelements = len(coeffOndeletteinter[i])
    for k in range(lengthelements):    # iteration over h the position
        deltatinet = detat/lengthelements 
        coeffOndelette.append((k+1/2)*deltatinet+int(t[0]))
sizescalefunction = coeffOndeletteinter[0].shape[0]

#### coefficients on our example
sizescalefunction = coeffOndeletteinter[0].shape[0]
NbCO = len(coeffOndelette) - sizescalefunction
sizeechel = len(coeffOndeletteinter[0])     # taille des coefficients d'echelle
NY = NbCO
Ymatrix = np.zeros( ( 1, NbCO))   # initialisation sur les donnée
j = 0   # ititialisation du dernier coefficient modifié
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  waveletH1[i].shape[0]  # On calcule le nombre de données disponibles
    Ymatrix[:,j:(j+tailleData)] = waveletH1[i]   # les coefficents de Haar sont ordonnées dans une matrix
    j = j+tailleData    # le dernier coefficient changé change

Y = Ymatrix.T.reshape(( NY,1))


############### Calculer X
#### Definition des entrées dans notre cas 
X = np.zeros( ( NbCO, dim+2))

jX = 0
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  coeffOndeletteinter[i].shape[0]  # On calcule le nombre de données disponibles
    X[jX:(jX+tailleData),1] = 2**(wlevel-i)#2**(i-1) # on positionnne les s
    for k in range( tailleData): # on positionne l'ensemble des u
        #X[(jX+k):((jX+(k+1))),2] =  coeffOndelette[j+sizescalefunction+k]*2**(wlevel-i)
        X[(jX+k):((jX+(k+1))),2] =  k #coeffOndelette[j+sizescalefunction+k]/2**(wlevel-i)#*2**(i-wlevel)
    #j = j+tailleData
    jX = jX+tailleData


#### image des distances entre points
distMatrix = np.zeros((NY,NY))
for i in range(NY):
    for j in range(NY):
        distMatrix[i,j] = np.abs(X[i,1]*(X[i,2]+1/2) - X[j,1]*(X[j,2]+1/2))

plt.imshow(distMatrix)
plt.colorbar()
plt.show()

### definition of the correlation length
l = 1/2
#### Computation of the variance
from KernelHaarMatern import KernHaarMatern52
kernelHaar = KernHaarMatern52(2+dim,dim, lengthscale=l*2**(M-1))

CovMatrix = kernelHaar.K(X)

#plt.show()

#### Compareason with the empirical matrix
import scipy.linalg as lng
dim = 1
MatKern = GPy.kern.Matern52(dim, lengthscale=l, variance=1.) # noyau
#GP = GPy.models.GPRegression(X=t[:2].reshape(2,1), Y=yH[:2].reshape(2,1), kernel=MatKern, noise_var=0)
#GP.randomize(np.random.normal)
#GP.set_XY()
#GP.Mat52.lengthscale.constrain_fixed(lengthscale)
#GP.Mat52.variance.constrain_fixed(1)

sizeGP = 100000
#postGen = GP.posterior_samples_f(t2.reshape(Nt2,1),size=sizeGP)
randomgrainecontinu = np.random.normal(0,1,(tcontinu.shape[0],sizeGP))
postGencontinu = lng.sqrtm(MatKern.K(tcontinu.reshape(Nt,1))+10**-14*np.diag(np.ones(Nt)))@ randomgrainecontinu
### On sous échantillonne la fonction
postGen = np.zeros((2**M, sizeGP))
for k in range(2**M):
    postGen[k] = np.sum(postGencontinu[k*2**N:(k+1)*2**N,:],0)* 1/2**N

# for i in range(20):
#     plt.plot(t,postGen[:,i])
# plt.show()
waveletH = pywt.wavedec(postGen.T, db1, mode='constant', level=wlevel)  # on the wavelet space
Ymatrix = np.zeros( ( sizeGP, NbCO))   # initialisation sur les donnée
j = 0   # ititialisation du dernier coefficient modifié
for i in range(1,wlevel+1):  # on fait l'intération sur les niveaux
    tailleData =  waveletH[i].shape[1]  # On calcule le nombre de données disponibles
    Ymatrix[:,j:(j+tailleData)] = waveletH[i]/np.sqrt(2)  # les coefficents de Haar sont ordonnées dans une matrix
    j = j+tailleData    # le dernier coefficient changé change

Mcov = np.cov((Ymatrix.T))

# #### Comparaison des coefficients d'ondelette avec la théorie
# # decompostion du dernier niveau
# postGeninter = postGen
# coeffinter = np.zeros((postGeninter.shape[0]//2,postGen.shape[1]))
# for i in range(coeffinter.shape[0]):
#     coeffinter[i,:] = (postGeninter[2*i,:] - postGeninter[2*i+1,:])/2
# postGeninter2 = np.zeros((postGeninter.shape[0]//2,postGen.shape[1]))
# for i in range(coeffinter.shape[0]):
#     postGeninter2[i,:] = (postGeninter[2*i,:] + postGeninter[2*i+1,:])
# postGeninter = postGeninter2/2
# waveletTheoryinter = [coeffinter]
# for i in range(1,wlevel):
#     coeffinter = np.zeros((postGeninter.shape[0]//2,postGen.shape[1]))
#     for i in range(coeffinter.shape[0]):
#         coeffinter[i,:] = (postGeninter[2*i,:] - postGeninter[2*i+1,:])
#     waveletTheoryinter.append(coeffinter/np.sqrt(2))
#     postGeninter2 = np.zeros((postGeninter.shape[0]//2,postGen.shape[1]))
#     for i in range(coeffinter.shape[0]):
#         postGeninter2[i,:] = (postGeninter[2*i,:] + postGeninter[2*i+1,:])
#     postGeninter = postGeninter2/np.sqrt(2)

# waveletTheory = [postGeninter]
# for i in range(wlevel):
#     waveletTheory.append(waveletTheoryinter[wlevel-i-1])

# #### Comparaison avec la boite pywt
# for i in range(wlevel+1):
#     print(i,np.mean(waveletH[i]/waveletTheory[i].T))
#     print(np.mean(waveletH[i]/np.sqrt(2)-waveletTheory[i].T))

fig = plt.figure()#figsize=(6, 4))
rows = 1
columns =3
fig.add_subplot(rows, columns, 1)
plt.title("Theorique")
plt.imshow((CovMatrix))
plt.colorbar()
fig.add_subplot(rows, columns, 2)
plt.title("Empirique")
plt.imshow((Mcov))
plt.colorbar()
fig.add_subplot(rows, columns, 3)
plt.title("Difference")
plt.imshow(abs((Mcov)-(CovMatrix))/abs(CovMatrix))
plt.colorbar()
plt.show()

#### We generate realization of the GP with the theorical matrix
randomwave = np.random.normal(0,1,(2**wlevel-1,sizeGP))
postGenOnde = lng.sqrtm(Mcov+10**-14*np.diag(np.ones(t.shape[0]-1))) @ randomwave
postGenOnde2 = lng.sqrtm(CovMatrix) @ randomwave
### We get back to the time space
GPHaarTemp =  [ np.random.normal(waveletH[0].mean(),waveletH[0].std(),(sizeGP,1))]#[np.zeros((sizeGP,1))] ## mean with normalisation of the coefficients
GPHaarTemp2 =  [np.random.normal(waveletH[0].mean(),waveletH[0].std(),(sizeGP,1))]
sumsetsize = 0  #### sum of the size
for i in range(1,wlevel+1): # iteration on the level
    setsize = coeffOndeletteinter[i].shape[0]   # size for this level
    inter = np.zeros( ( sizeGP, setsize))
    inter2 = np.zeros( ( sizeGP, setsize))
    for j in range(setsize):
        inter[:,j] = postGenOnde[sumsetsize:(sumsetsize+1),:].T[:,0]
        inter2[:,j] = postGenOnde2[sumsetsize:(sumsetsize+1),:].T[:,0]
        sumsetsize = sumsetsize + 1
    GPHaarTemp.append(inter*np.sqrt(2))
    GPHaarTemp2.append(inter2*np.sqrt(2))

GPHaarReal = pywt.waverec(GPHaarTemp, db1, mode='constant')  # realisation of GP Matern 5/2
GPHaarReal2 = pywt.waverec(GPHaarTemp2, db1, mode='constant')  # realisation of GP Matern 5/2
for i in range(2000):
    #plt.plot(t, GPHaarReal2[i,:],'b')
    plt.plot(t, GPHaarReal[i,:],'r')
    #plt.plot(t,postGen[:,i], 'r')

#plt.plot(t, np.std(GPHaarReal2,0),'r')
plt.show()

#### On affiche la covariance dans le temps de la fonction
fig = plt.figure()#figsize=(6, 4))
rows = 1
columns =3
fig.add_subplot(rows, columns, 1)
plt.title("Theorique")
plt.imshow(np.cov(GPHaarReal2.T))
plt.colorbar()
fig.add_subplot(rows, columns, 2)
plt.title("Empirique")
plt.imshow(np.cov(GPHaarReal.T))
plt.colorbar()
fig.add_subplot(rows, columns, 3)
plt.title("Cible")
plt.imshow(MatKern.K(t.reshape(2**M,1)))
plt.colorbar()
plt.show()

#### On cherche à comprendre la génération de nombre aléatoires
# randomwave2 = np.random.normal(0,1,(2**M,sizeGP))
# postGenOndeReal = lng.sqrtm(MatKern.K(t.reshape(2**M,1))+10**-14*np.diag(np.ones(t.shape[0]))) @ randomwave2

# for i in range(2000):
#     plt.plot(t, postGenOndeReal[:,i],'b')
#     #plt.plot(t,postGen[:,i], 'r')

# #plt.plot(t, np.std(GPHaarReal2,0),'r')
# plt.show()
## Le problème ne vient pas de là

### Les GP semblent être non stationnaire, en particulité aux bords il y a un problème
### Dans cette section je cherche à comprendre pourquoi
GPHaarTesTemps =  [waveletH[0]]
sumsetsize = 0  #### sum of the size
for i in range(1,wlevel+1): # iteration on the level
    setsize = coeffOndeletteinter[i].shape[0]   # size for this level
    inter = np.zeros( ( sizeGP, setsize))
    for j in range(setsize):
        inter[:,j] = Ymatrix[:,sumsetsize:(sumsetsize+1)][:,0]
        sumsetsize = sumsetsize + 1
    GPHaarTesTemps.append(inter*np.sqrt(2))

GPHaarTesReal = pywt.waverec(GPHaarTesTemps, db1, mode='constant')  # realisation of GP Matern 5/2
for i in range(2000):
    plt.plot(t, GPHaarTesReal[i,:],'b')
    #plt.plot(t,postGen[:,i], 'r')

#plt.plot(t, np.std(GPHaarReal2,0),'r')
plt.show()

##### Avec ce qui précède on peut montrer que la non stationnarité ne vient pas de la boite !
