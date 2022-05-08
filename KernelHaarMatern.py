#### Import 
from GPy.kern.src.kern import Kern
import numpy as np
from GPy.core.parameterization import Param

class KernHaarMatern52(Kern):
    """ Cette classe est celle définisant le noyau de covariance dans l'espace des 
    coefficients d'ondelette. C'est une classe fille de la classe Kern de GPy."""
    def __init__(self, input_dim=2, dim_start=1, lengthscale=1., active_dims=None):
        super(KernHaarMatern52, self).__init__(input_dim, active_dims, 'haar_m52')
       # assert 
        self.dim_start = dim_start
        self.lengthscale = Param('lengthscale', lengthscale)
        #self.power = Param('power', power)
        self.link_parameters( self.lengthscale)#, self.power)

    #### fonction pour annoncer les changements de paramètres
    def parameters_changed(self):
        ### rien à faire
        pass

    #### La fonction est utiliser pour calculer la covariance entre X1 et X2
    def K( self, X1, X2):
        if X2 is None: X2=X1
        #inter = np.zeros((X1.shape[0], X2.shape[0]))
        #for i in range(X1.shape[0]):
        #    inter[i,:] = self.coeffInteger( X1[i,self.dim_start], X2[:,self.dim_start])/self.lengthscale**5 *self.integralFull( X1[i,self.dim_start+1], X2[:,self.dim_start+1], X1[i,self.dim_start], X2[:,self.dim_start])
        inter = self.coeffInteger( X1[:,self.dim_start], X2[:,self.dim_start]) *self.integralFull( X1[:,self.dim_start+1], X2[:,self.dim_start+1], X1[:,self.dim_start], X2[:,self.dim_start])
        return(  inter)

    ### la méthode pour caculer la varianceX1[:,self.dim_start+1]
    def Kdiag( self, X):
        return( self.coeffInteger( X[:,self.dim_start], X[:,self.dim_start]) *self.integralFull( X[:,self.dim_start+1], X[:,self.dim_start+1], X[:,self.dim_start], X[:,self.dim_start]))


    ### la méthode pour calculer le coefficient de normalisation de la covariance
    def coeffInteger( self,  s, sstar):
        """" On calcule le coefficient de normalisation de l'intégralle"""
        ### Ici il y a un petit problème de taille avec s et sstar 
        si =s.reshape(s.shape[0],1)
        sistar =sstar.reshape(sstar.shape[0],1)
        return( (200*np.sqrt(5)*16)/(3*(np.sqrt(si*sistar.T))*np.pi))    ### Correction by the $2\pi$

    ### La méthode pour calculer la covariance totale
    def integralFull( self, u, ustar, s, sstar):
        """ On calcule l'intégral pour le noyau de covariance. 
            On dispose de la formule exact de l'integral """

        ## Definition des alpha_i
        alpha = np.array([ 1/4, -1/8, -1/8, -1/8, -1/8, 1/16, 1/16, 1/16, 1/16]).reshape(9,1,1)
        #### Generation de gamma_i
        s = s.reshape(s.shape[0],1)
        u = u.reshape(s.shape[0],1)
        sstar = sstar.reshape(sstar.shape[0],1)
        ustar = ustar.reshape(sstar.shape[0],1)
        ### defintion de la différence des gamma
        gamma = np.array([ np.kron(s, sstar.T*0+1)*0/2+np.kron(s*0+1, sstar.T)*0/2, np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2*0, -np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2*0, np.kron(s, sstar.T*0+1)*0/2-np.kron(s*0+1, sstar.T)/2,  np.kron(s, sstar.T*0+1)*0/2+np.kron(s*0+1, sstar.T)/2,  np.kron(s, sstar.T*0+1)/2-np.kron(s*0+1, sstar.T)/2, -np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2, np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2, -np.kron(s, sstar.T*0+1)/2-np.kron(s*0+1, sstar.T)/2])
        ### on ajoute le terme constant dans le gamma
        gamma = gamma + np.repeat((np.kron(s*(u+1/2), sstar.T*0+1)-np.kron(s*0+1, sstar.T*(ustar.T+1/2)))[ np.newaxis,:,:],9,axis=0)
        ### On calcule le terme de la somme de fraction
        termSum = ( 3* self.lengthscale**2/(200*np.sqrt(5)) + 7* np.abs(gamma)*self.lengthscale/1000 + np.sqrt(5)*gamma**2/(1000))
        ### On calcule le terme exponetiel
        ProdTerm = np.pi * np.exp(-np.sqrt(5)*np.abs(gamma)/self.lengthscale)
        ### On réalise la somme
        return( - np.sum(alpha * ProdTerm * termSum, 0)  - np.pi/125*np.sum(alpha* np.abs(gamma),0)*self.lengthscale )

    ### La méthode pour calculer la dérivée de la covariance totale
    def integralFullDerivative( self, u, ustar, s, sstar, Npoints = 100, epsilon=10**-6):
        """ On calcule l'intégral de la dérivée du noyau de covariance. 
            Pour cela on dispose d'une formule analytique."""
        ## Definition des alpha_i
        alpha = np.array([ 1/4, -1/8, -1/8, -1/8, -1/8, 1/16, 1/16, 1/16, 1/16]).reshape(9,1,1)
        #### Generation de gamma_i
        s = s.reshape(s.shape[0],1)
        u = u.reshape(s.shape[0],1)
        sstar = sstar.reshape(sstar.shape[0],1)
        ustar = ustar.reshape(sstar.shape[0],1)
        ### defintion de la différence des gamma
        gamma = np.array([ np.kron(s, sstar.T*0+1)*0/2+np.kron(s*0+1, sstar.T)*0/2, np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2*0, -np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2*0, np.kron(s, sstar.T*0+1)*0/2-np.kron(s*0+1, sstar.T)/2,  np.kron(s, sstar.T*0+1)*0/2+np.kron(s*0+1, sstar.T)/2,  np.kron(s, sstar.T*0+1)/2-np.kron(s*0+1, sstar.T)/2, -np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2, np.kron(s, sstar.T*0+1)/2+np.kron(s*0+1, sstar.T)/2, -np.kron(s, sstar.T*0+1)/2-np.kron(s*0+1, sstar.T)/2])
        ### on ajoute le terme constant dans le gamma
        gamma = gamma + np.repeat((np.kron(s*(u+1/2), sstar.T*0+1)-np.kron(s*0+1, sstar.T*(ustar.T+1/2)))[ np.newaxis,:,:],9,axis=0)
        ### On calcule le premier terme de la somme de fraction
        termSum1 = ( 3* self.lengthscale/(100*np.sqrt(5)) + 7* np.abs(gamma)/1000)
        ### On calcule le deuxième terme de la somme de fraction
        termSum2 = ( 3* self.lengthscale/(200*np.sqrt(5)) + 7* np.abs(gamma)/1000 + np.sqrt(5)*gamma**2/(1000*self.lengthscale))
        ### On calcule le premier terme exponetiel
        ProdTerm1 = alpha * np.pi * np.exp(-np.sqrt(5)*np.abs(gamma)/self.lengthscale)
        ### On calcule le deuxième terme exponetiel
        ProdTerm2 = alpha * np.pi * np.sqrt(5)*np.abs(gamma)/self.lengthscale**2 * np.exp(-np.sqrt(5)*np.abs(gamma)/self.lengthscale)
        ### On réalise la somme
        return( - np.sum(ProdTerm1*termSum1 - ProdTerm2*termSum2, 0)  - np.pi/125*np.sum(alpha* np.abs(gamma),0) )
    
    #### la fonction pour updater le gradient
    def update_gradients_full(self, dL_dK, X1, X2):
        if X2 is None: X2 = X1
        
        self.lengthscale.gradient = np.sum(dL_dK *self.coeffInteger( X1[:,self.dim_start], X2[:,self.dim_start]) *self.integralFullDerivative( X1[:,self.dim_start+1], X2[:,self.dim_start+1], X1[:,self.dim_start], X2[:,self.dim_start]))