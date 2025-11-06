#%%
#Aplicación de MCMC MS en la funcion luminocidad

import numpy as np
import pandas as pd

tb=pd.read_csv('datos_Blanton.csv')
M=tb['MAG']
vfl=tb['PHI']
er_inf=tb['error_inf']
er_sup=tb['error_sup']
sigma=(er_inf+er_sup)/2
phim=(0.014,-21,-1.2)
phim=np.array(phim)
alfa=(0.0001,0.01,0.001)
alfa=np.array(alfa)

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

def FL(x, phi):
    '''Función de Schechter

    Parámetros:
    x: variable independiente de la muestra.
    phi: lista de parámetros.
    '''
    FL=0.4*np.log(10)*phi[0]*10**(-0.4*(x-phi[1])*(phi[2]+1))*np.exp(-10**(-0.4*(x-phi[1])))
    return FL


def logprior(x,y,phi):
    '''Logaritmo natural del prior normal con sigma igual al promedio de error inferior y superior de cada medición.
    
    Parámetros:
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    phi: lista de parámetros.
    
    Resultado
    logL: suma sobre los pares de coordenadas de logaritmo natural del prior.'''
    gtot=0
    for i in range(len(phi)):
        gtot=gtot+np.log(np.exp(-0.5*(phi[i]-phim[i])**2/(alfa[i]*100)**2))
    return gtot

def logposterior(x,y,f_mod,logprior,phi):
    '''Logaritmo natural del posterior

    Parámetros:
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    f_mod: función modelo.
    logprior: función de asignación de priors.
    phi: lista de parámetros.
    
    Resultado:
    logL: logaritmo natural de la probabilidad posterior sumada sobre los pares (x,y).'''
    logPost=sum(loglike_gaussiano(x,y,f_mod,phi)+logprior(x,y,phi))
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
    m=len(phi0)
    phi=np.random.uniform(-1,1,size=m)*alpha+phi0
    phi_m=[]
    phi_m.append(phi)
    for i in range(n):
        phi_n=phi+alpha*np.random.uniform(-1,1,m)
        delta=logposterior(x,y,f_mod,logprior,phi_n)-logposterior(x,y,f_mod,logprior,phi)
        if delta>0 or np.exp(delta)>np.random.uniform(0,1):
                phi=phi_n.copy()


        phi_m.append(phi.copy())
        
                
    return pd.DataFrame(phi_m,columns=list(range(m)))

phi_fl=MCMC_MH(M,vfl,FL,alfa,logprior,phim,5000)
#%%
#Visualización de los resultados
import seaborn as sns
import matplotlib.pyplot as plt
c1='mediumseagreen'
c2='purple'
c3='darkorange'

corte=1
phi=phi_fl.iloc[corte:,0]
Mc=phi_fl.iloc[corte:,1]
alfac=phi_fl.iloc[corte:,2]
i=np.linspace(1,len(phi),len(phi))

plt.scatter(i,phi,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel('$\phi*$')
plt.show()
plt.scatter(i,Mc,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel('$M*$')
plt.show()
plt.scatter(i,alfac,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel(r'$\alpha$')
plt.show()

corte=3000
phi_c=phi_fl.iloc[corte:,0]
Mc_c=phi_fl.iloc[corte:,1]
alfac_c=phi_fl.iloc[corte:,2]
i_c=np.linspace(1,len(phi),len(phi))

phi_m=phi_c.mean()
phi_st=phi_c.std()
Mc_m=Mc_c.mean()
Mc_st=Mc_c.std()
alfac_m=alfac_c.mean()
alfac_st=alfac_c.std()

rango=np.linspace(-23.3,-16,100)
#
plt.plot(rango,FL(rango,phim),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(M,vfl-er_inf,vfl+er_sup,colors=c1,label='Barras de error')
plt.scatter(M,vfl,c=c1,label='Muestra')
plt.plot(rango,FL(rango,[phi_m,Mc_m,alfac_m]),color=c2,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_fl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['$\phi*$', 'M*','flfa']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()





#%%
#Método del gradiente descendente
import numpy as np
import pandas as pd



def grad_lum(x,y,p):
    '''
    Cálculo de los gradientes por parámetro del Likelyhood, asumiendo errores normales y un modelo FL.
    
    Parámetros:
    x: variable indepentendiente de la muestra.
    y: variable dependiente de la muestra.
    p: paámetros.

    Resultado:
    Lista de los gradientes por paámetro normalizados.
    '''
    FL_val=FL(x,p)
    L=np.exp(loglike_gaussiano(x,y,FL,p))
    dLdf=(y-FL_val)/(sigma**2)
    devphi=1/p[0] #El gradiente tiene mas coeficientes, pero al normalizarlos no son necesarios
    devM=np.log(10)*0.4*(1+p[2]-10**(-0.4*(x-p[1])))
    deva=-np.log(10)*0.4*(x-p[1])
    grad=(devphi*len(x),sum(devM),sum(deva))
    grad=np.array(grad)
    grad=grad/sum(grad)
    return grad

def pasos_gradiente(x,y,grad,p0,alpha,n=5000):
    '''Calculo de los pasos 
    
    Parámetros:
    x: variable indepentendiente de la muestra.
    y: variable dependiente de la muestra.
    grad: lista de derivadas del likelihood respecto a los paámetros, normalizada.
    p0: lista de paámetros estimados.
    alpha: tamaño de los pasos por paámetro.
    n=500: número de pasos.

    Resultado:
    Pandas DataFrame con columnas respectivas a cada pámetro y filas respectivas a sus valores en cada paso.
    '''
    m=len(p0)
    p0=np.array(p0)
    alpha=np.array(alpha)
    p=p0+alpha*np.random.uniform(-1,1,size=m)
    tb_p=[]
    for i in range(n):
        grad=np.array(grad_lum(x,y,p))
        p_n=p+np.random.uniform(-1,1,m)*alpha*grad
        p=p_n
        tb_p.append(p)
    return pd.DataFrame(tb_p,columns=list(range(m)))

tb_flgrad=pasos_gradiente(M,vfl,grad=grad_lum,p0=phim,alpha=alfa,n=5000)

#%%
#Visualización de los resultados
import seaborn as sns
import matplotlib.pyplot as plt
c1='mediumseagreen'
c2='purple'
c3='darkorange'

corte=2000
phi=tb_flgrad.iloc[corte:,0]
Mc=tb_flgrad.iloc[corte:,1]
alfac=tb_flgrad.iloc[corte:,2]
i=np.linspace(1,len(phi),len(phi))

plt.scatter(i,phi,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel('$\phi*$')
plt.show()
plt.scatter(i,Mc,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel('$M*$')
plt.show()
plt.scatter(i,alfac,c=c2,s=1)
plt.xlabel('Pasos')
plt.ylabel(r'$\alpha$')
plt.show()


phi_m=phi.mean()
phi_st=phi.std()
Mc_m=Mc.mean()
Mc_st=Mc.std()
alfac_m=alfac.mean()
alfac_st=alfac.std()

rango=np.linspace(-23.3,-16,100)
#
plt.plot(rango,FL(rango,phim),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(M,vfl-er_inf,vfl+er_sup,colors=c1,label='Barras de error')
plt.scatter(M,vfl,c=c1,label='Muestra')
plt.plot(rango,FL(rango,[phi_m,Mc_m,alfac_m]),color=c2,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = tb_flgrad
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['$\phi*$', 'M*','flfa']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()

