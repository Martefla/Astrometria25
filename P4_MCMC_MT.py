#%%
#Generación de la muestra:
import matplotlib.pyplot as plt
import numpy as np


c1='mediumseagreen'
c2='purple'
c3='darkorange'
n=50
rango=(0,10)
sigma=4
x=np.linspace(rango[0],rango[1],n)
e=np.random.normal(size=n,scale=sigma)
a_0=5
b_0=3
y=b_0+a_0*x+e


#%%

#Definición de la función pasos general:

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
    Pandas DataFrame con columnas respectivas a cada pámetro y filas respectivas a sus valores en cada paso.
    Número de veces que un paso fue aceptado'''
    m=len(phi0)
    phi=np.random.uniform(-1,1,size=m)*alpha*10+phi0
    phi_m=[]
    phi_m.append(phi)
    aceptacion=0
    for i in range(n):
        phi_n=phi+alpha*np.random.uniform(-1,1,m)
        delta=logposterior(x,y,f_mod,logprior,phi_n)-logposterior(x,y,f_mod,logprior,phi)
        if delta>0 or delta>np.log(np.random.random()):
            phi=phi_n.copy()
            aceptacion=aceptacion + 1

        phi_m.append(phi.copy())
        
                
    return pd.DataFrame(phi_m,columns=list(range(m))), aceptacion

#%%
#Definición de las componentes bayesianas para este problema.
import numpy as np
import pandas as pd
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


def logprior_pl(x,y,phi):
    '''Logaritmo natural del prior plano en los rangos [4,6] y [2,4].
    
    phi: lista de parámetros.
    
    Resultado
    
    logL: logaritmo natural del prior.'''
    if ((phi[0]>4) & (phi[0]<6) & (phi[1]>2) & (phi[1]<4)):
        g=0
    else:
        g=-np.inf
    return g


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

#Aplicación de la MCMC a los datos lineales 

def lin(x,phi):
    lin=phi[0]*x+phi[1]
    return lin

alpha=(0.01,0.01)
alpha=np.array(alpha)
phi0=(5,3)

phi_pl,acep_pl=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_pl,alpha=alpha,phi0=phi0,n=5000)

#MCMC con priors gaussianos

def logprior_gauss(x,y,phi):
    '''Logaritmo natural del prior normal con sigma 4.
    
    Parámetros:
    x: variable independiente de la muestra.
    y: variable dependiente de la muestra.
    phi: lista de parámetros.
    
    Resultado
    logL: suma sobre los pares de coordenadas de logaritmo natural del prior.'''
    g=sum(np.log(np.exp(-0.5*(y-lin(x,phi))**2/4**2)))
    return g

phi_gauss,acep_gs=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_gauss,alpha=alpha,phi0=phi0,n=5000)

#%%
#Visualización de los datos
import seaborn as sns
i=np.linspace(1,len(phi_pl[0]),len(phi_pl[0]))
j=np.linspace(1,len(phi_gauss[0]),len(phi_gauss[0]))
corte=1000
a_pl=phi_pl.iloc[:,0]
b_pl=phi_pl.iloc[:,1]
a_gs=phi_gauss.iloc[:,0]
b_gs=phi_gauss.iloc[:,1]

plt.scatter(i,a_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.scatter(j,a_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.legend()
plt.show()

plt.scatter(i,b_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.scatter(j,b_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.legend()
plt.show()

a_mpl=a_pl.mean()
a_stpl=a_pl.std()
b_mpl=b_pl.mean()
b_stpl=b_pl.std()

a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
a_mgs=a_gs.mean()
a_stgs=a_gs.std()
b_mgs=b_gs.mean()
b_stgs=b_gs.std()


plt.plot(rango,(rango[0]*a_0+b_0,rango[1]*a_0+b_0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error a un $\sigma$')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,(rango[0]*a_mpl+b_mpl,rango[1]*a_mpl+b_mpl),color=c2,linewidth=2,label='Función MCMC de prior plano')
plt.plot(rango,(rango[0]*a_mgs+b_mgs,rango[1]*a_mgs+b_mgs),color=c3,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend(loc='upper left')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_pl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()


sns.set_theme(palette=[c3])
df = phi_gauss
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)



#%%
#Estudio con diferente longitud de pasos
alpha=(0.0001,0.0001)

phi_pl,acep_pl=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_pl,alpha=alpha,phi0=phi0,n=5000)
phi_gauss,acep_gs=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_gauss,alpha=alpha,phi0=phi0,n=5000)
#Visualización de los datos
import seaborn as sns

corte=1000
a_pl=phi_pl.iloc[corte:,0]
b_pl=phi_pl.iloc[corte:,1]
a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
i=np.linspace(1,len(a_pl),len(a_pl))
j=np.linspace(1,len(a_gs),len(a_gs))

plt.scatter(i,a_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.scatter(j,a_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.legend()
plt.show()

plt.scatter(i,b_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.scatter(j,b_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.legend()
plt.show()

a_mpl=a_pl.mean()
a_stpl=a_pl.std()
b_mpl=b_pl.mean()
b_stpl=b_pl.std()

a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
a_mgs=a_gs.mean()
a_stgs=a_gs.std()
b_mgs=b_gs.mean()
b_stgs=b_gs.std()

#
plt.plot(rango,(rango[0]*a_0+b_0,rango[1]*a_0+b_0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error a un $\sigma$')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,(rango[0]*a_mpl+b_mpl,rango[1]*a_mpl+b_mpl),color=c2,linewidth=2,label='Función MCMC de prior plano')
plt.plot(rango,(rango[0]*a_mgs+b_mgs,rango[1]*a_mgs+b_mgs),color=c3,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_pl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()


sns.set_theme(palette=[c3])
df = phi_gauss
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)

print('Tasa de aceptación del prior plano:',acep_pl)
print('Tasa de aceptación del prior normal:',acep_gs)

#%%
#Estudio con diferente longitud de pasos
alpha=(0.1,0.1)

phi_pl,acep_pl=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_pl,alpha=alpha,phi0=phi0,n=5000)
phi_gauss,acep_gs=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_gauss,alpha=alpha,phi0=phi0,n=5000)
#Visualización de los datos
import seaborn as sns

corte=1000
a_pl=phi_pl.iloc[corte:,0]
b_pl=phi_pl.iloc[corte:,1]
a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
i=np.linspace(1,len(a_pl),len(a_pl))
j=np.linspace(1,len(a_gs),len(a_gs))

plt.scatter(i,a_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.scatter(j,a_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.legend()
plt.show()

plt.scatter(i,b_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.scatter(j,b_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.legend()
plt.show()

a_mpl=a_pl.mean()
a_stpl=a_pl.std()
b_mpl=b_pl.mean()
b_stpl=b_pl.std()

a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
a_mgs=a_gs.mean()
a_stgs=a_gs.std()
b_mgs=b_gs.mean()
b_stgs=b_gs.std()

#
plt.plot(rango,(rango[0]*a_0+b_0,rango[1]*a_0+b_0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error a un $\sigma$')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,(rango[0]*a_mpl+b_mpl,rango[1]*a_mpl+b_mpl),color=c2,linewidth=2,label='Función MCMC de prior plano')
plt.plot(rango,(rango[0]*a_mgs+b_mgs,rango[1]*a_mgs+b_mgs),color=c3,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_pl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()


sns.set_theme(palette=[c3])
df = phi_gauss
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
print('Tasa de aceptación del prior plano:',acep_pl)
print('Tasa de aceptación del prior normal:',acep_gs)
#%%
#Estudio con diferente cantidad de pasos
alpha=(0.01,0.01)

phi_pl,acep_pl=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_pl,alpha=alpha,phi0=phi0,n=50000)
phi_gauss,acep_gs=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_gauss,alpha=alpha,phi0=phi0,n=50000)
#Visualización de los datos
import seaborn as sns

corte=10000
a_pl=phi_pl.iloc[corte:,0]
b_pl=phi_pl.iloc[corte:,1]
a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
i=np.linspace(1,len(a_pl),len(a_pl))
j=np.linspace(1,len(a_gs),len(a_gs))

plt.scatter(i,a_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.scatter(j,a_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.legend()
plt.show()

plt.scatter(i,b_pl,c=c2,s=1,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.scatter(j,b_gs,c=c3,s=1,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.legend()
plt.show()

a_mpl=a_pl.mean()
a_stpl=a_pl.std()
b_mpl=b_pl.mean()
b_stpl=b_pl.std()

a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
a_mgs=a_gs.mean()
a_stgs=a_gs.std()
b_mgs=b_gs.mean()
b_stgs=b_gs.std()

#
plt.plot(rango,(rango[0]*a_0+b_0,rango[1]*a_0+b_0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error a un $\sigma$')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,(rango[0]*a_mpl+b_mpl,rango[1]*a_mpl+b_mpl),color=c2,linewidth=2,label='Función MCMC de prior plano')
plt.plot(rango,(rango[0]*a_mgs+b_mgs,rango[1]*a_mgs+b_mgs),color=c3,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_pl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()


sns.set_theme(palette=[c3])
df = phi_gauss
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)

print('Tasa de aceptación del prior plano:',acep_pl*100/n)
print('Tasa de aceptación del prior normal:',acep_gs*100/n)

#%%
#Estudio con diferente cantidad de pasos
alpha=(0.01,0.01)
n=500

phi_pl,acep_pl=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_pl,alpha=alpha,phi0=phi0, n=n)
phi_gauss,acep_gs=MCMC_MH(x=x,y=y,f_mod=lin,logprior=logprior_gauss,alpha=alpha,phi0=phi0,n=n)
#Visualización de los datos
import seaborn as sns
corte=100
a_pl=phi_pl.iloc[corte:,0]
b_pl=phi_pl.iloc[corte:,1]
a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
i=np.linspace(1,len(a_pl),len(a_pl))
j=np.linspace(1,len(a_gs),len(a_gs))

plt.plot(i,a_pl,color=c2,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.plot(j,a_gs,color=c3,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('a')
plt.legend()
plt.show()

plt.plot(i,b_pl,color=c2,label='Prior plano')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.plot(j,b_gs,color=c3,label='Prior normal')
plt.xlabel('Pasos')
plt.ylabel('b')
plt.legend()
plt.show()

a_mpl=a_pl.mean()
a_stpl=a_pl.std()
b_mpl=b_pl.mean()
b_stpl=b_pl.std()

a_gs=phi_gauss.iloc[corte:,0]
b_gs=phi_gauss.iloc[corte:,1]
a_mgs=a_gs.mean()
a_stgs=a_gs.std()
b_mgs=b_gs.mean()
b_stgs=b_gs.std()


plt.plot(rango,(rango[0]*a_0+b_0,rango[1]*a_0+b_0),color='k',alpha=0.5,linewidth=3,label='Función generativa')
plt.vlines(x,y-sigma/2,y+sigma/2,colors=c1,label='Barras de error a un $\sigma$')
plt.scatter(x,y,c=c1,label='Muestra')
plt.plot(rango,(rango[0]*a_mpl+b_mpl,rango[1]*a_mpl+b_mpl),color=c2,linewidth=2,label='Función MCMC de prior plano')
plt.plot(rango,(rango[0]*a_mgs+b_mgs,rango[1]*a_mgs+b_mgs),color=c3,linewidth=2,label='Función MCMC de prior gaussiano')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

sns.set_theme(palette=[c2])
df = phi_pl
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)
plt.show()


sns.set_theme(palette=[c3])
df = phi_gauss
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

ejes=['Pendiente', 'Ordenada al origen']

for i, lab in enumerate(ejes):
    g.axes[-1, i].set_xlabel(lab)  
    g.axes[i, 0].set_ylabel(lab)

print('Tasa de aceptación del prior plano:',acep_pl*100/n)
print('Tasa de aceptación del prior normal:',acep_gs*100/n)