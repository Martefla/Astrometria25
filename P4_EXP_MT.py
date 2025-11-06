#%%
#MCMC MH para una distribución exponencial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
n=50
sigma=0.01
x=np.linspace(0,10,n)
e=np.random.normal(size=n,scale=sigma)
lam0=0.2
lam0=np.array(lam0)
y=lam0*np.exp(-lam0*x)+e

def loglike_gaussiano(x, y, f, phi):
    '''Logaritmo natural del likelihood con errores gaussianos
    
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    phi: lista de parámetros.
    
    Resultado
    
    logL: logaritmo natural del likelihood.'''
    residuals = y - f(x, phi)
    n = len(y)
    logL = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2
    return logL

def logposterior(x,y,f_mod,logprior,phi):
    '''Logaritmo natural del posterior

    Parámetros:
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    f_mod: función modelo.
    logprior: función de asignación de priors.
    phi: lista de parámetros.
    
    Resultado:
    logL: logaritmo natural de la probabilidad posterior.'''
    logPost=loglike_gaussiano(x,y,f_mod,phi)+logprior(x,y,phi)
    return logPost

def MCMC_MH(x,y,f_mod,alpha,logprior,phi0,n=500):
    '''Cálculo de los pasos en una cadena de makov con el método de  Metrópolis Hastings.

    Parámetros:
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    f_mod: función modelo.
    alpha: lista del tamaño de los pasos para cada uno de los parámetros.
    logprior: función de asignación de priors.
    phim: lista de parámetros estimados.
    
    Resultado:
    Pandas DataFrame con columnas respectivas a cada pámetro y filas respectivas a sus valores en cada paso.'''
    m=1
    phi=np.random.uniform(-1,1,size=m)*alpha+phi0
    phi_m=[]
    phi_m.append(phi)
    for i in range(n):
        phi_n=phi+alpha*np.random.uniform(-1,1,m)
        delta=logposterior(x,y,f_mod,logprior,phi_n)-logposterior(x,y,f_mod,logprior,phi)
        if delta>0 or delta>np.log(np.random.random()):
            phi=phi_n.copy()

        phi_m.append(phi.copy())
        
                
    return pd.DataFrame(phi_m,columns=list(range(m)))

#Definiciones para utilizar MCMC
err=sigma
alpha=(0.01)
alpha=np.array(alpha)

def exp(x,lam):
    exp=lam*np.exp(-lam*x)
    return exp

def logprior(x,y,phi):
    g=sum(np.log(np.exp(-0.5*(y-exp(x,phi))**2/err**2)))
    return g

tb_lam=MCMC_MH(x,y,exp,alpha,logprior,lam0,5000)

#%%
#Calculo analítico de lambda
def minlike(lam_op):
    minlike=sum((np.exp(-lam_op*x)*y-(np.exp(-2*lam_op*x))*lam_op)*(1-x*lam_op))
    return minlike
from scipy.optimize import root_scalar

res = root_scalar(minlike,bracket=[-0.5,0.5],method='brentq')
print(res)
#%%
#Visualización de los resultados
import seaborn as sns
import matplotlib.pyplot as plt
c1='mediumseagreen'
c2='purple'
c3='darkorange'

corte=1000
lam=tb_lam.iloc[corte:,0]

i=np.linspace(1,len(lam),len(lam))

plt.plot(i,lam,c=c2)
plt.xlabel('Pasos')
plt.ylabel('$\lambda$')
plt.show()

lam_m=lam.mean()
lam_st=lam.std()

rango=np.linspace(0,10,100)

plt.plot(rango,exp(rango,lam0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,exp(rango,lam_m),color=c2,linewidth=2,label='Función MCMC de prior gaussiano')
plt.plot(rango,exp(rango,0.1995),color=c3,linewidth=2,label='Función minimizando likelihood')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.hist(lam,color=c1,density=True,bins=30)
plt.xlabel('$\lambda$')
plt.show()