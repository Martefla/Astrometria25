#%%
#%% MCMC MH aplicado en likelihood binomial
import numpy as np
from scipy.stats import binom
caras=60
secas=40
n=100
p0=0.6
y=np.random.binomial(n,p0)
p=np.linspace(0,1,100)

def logprior_monpl(p):
    if p > 0.25 and p < 0.95:
        logg=0
    else:
        logg=-np.inf
    return logg

def loglike_bin(y,p):
    logl=np.log(binom.pmf(y,n,p))
    return logl

def logpost_mon(y,p):
    logp=logprior_monpl(p)+loglike_bin(y,p)
    return logp

logp=[]
for i in range(len(p)):
    logp.append(logpost_mon(y,p[i]))

#Visualización de los datos
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt

c1='purple'
c2='mediumseagreen'

def gaussiana(y,mu,sigma,A):
    gauss=np.exp(-(y-mu)**2/(2*sigma**2))*A
    return gauss
p_n=p[30:]
post=np.exp(logp[30:])
par, cov = curve_fit(gaussiana,p_n,post,p0=[0.6,0.2,0.1])

sns.scatterplot(x=p_n,y=post,color=c1,label='Valores de la probabilidad posterior')
sns.lineplot(x=p_n,y=gaussiana(p_n,par[0],par[1],par[2]),color=c2,label='Gaussiana ajustada')

plt.xlabel('Sesgo')
plt.ylabel('Probabilidad posterior')

print('Valor del sesgo=',par[0])

#%%
#Prior Gaussiano

def logprior_mongs(p):
    logg=np.log(np.exp(-0.5*(p-0.6)**2/0.1**2))
    return logg

def loglike_bin(y,p):
    logl=np.log(binom.pmf(y,n,p))
    return logl

def logpost_mon(y,p):
    logp=logprior_mongs(p)+loglike_bin(y,p)
    return logp

logp=[]
for i in range(len(p)):
    logp.append(logpost_mon(y,p[i]))

#Visualización de los datos
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt

c1='purple'
c2='mediumseagreen'

def gaussiana(y,mu,sigma,A):
    gauss=np.exp(-(y-mu)**2/(2*sigma**2))*A
    return gauss
p_n=p[30:]
post=np.exp(logp[30:])
par, cov = curve_fit(gaussiana,p_n,post,p0=[0.6,0.2,0.1])

sns.scatterplot(x=p_n,y=post,color=c1,label='Valores de la probabilidad posterior')
sns.lineplot(x=p_n,y=gaussiana(p_n,par[0],par[1],par[2]),color=c2,label='Gaussiana ajustada')

plt.xlabel('Sesgo')
plt.ylabel('Probabilidad posterior')

print('Valor del sesgo=',par[0])