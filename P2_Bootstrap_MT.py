#%%
#Remuestreo Bootstrap.
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

def bootstrap(x,func,m=1000):
    '''
    Remuestreo bootstrap para calcular un estadístico de la muestra.

    Prámetros:
    x: lista de los valores de la muestra.
    func: estadítico que se quiera calcular en forma de función.
    m=1000: número de submuestras para el cálculo.

    Returns:
    y: valor del estadítico indicado de bootstrap.
    '''
    y=np.zeros(m)
    for i in range(m):
        _x = np.random.choice(x,size=len(x),replace=True)
        y[i]=func(_x)
    return y

#%%
import numpy as np
import scipy.stats as st
from fun import bootstrap

def IC(muestra,func,alfa=0.05,n=100):
    '''
    Calculo de intevalos de confianza para un estaditico utilizando remuestreo de bootstrap.

    Parámetros:
    muestra: lista de los valores de la muestra.
    func: estadístico al que calcularle el intevalo de confianza.
    alfa=0.05: valor de significancia, entre 0 y 1.
    n=100: número de submuestras para el cálculo.

    Return:
    IC1: valor inferior del intervalo de confianza.
    IC2: valor superior del intervalo de confianza.

    '''
    m=len(muestra)
    y=bootstrap(muestra,func,n)
    z1=st.norm.ppf(alfa/2,loc=0.0,scale=1.0)
    z2=st.norm.ppf(1-alfa/2,loc=0.0,scale=1.0)
    IC2=y.mean()+z2*(y.std())
    IC1=y.mean()+z1*(y.std())
    return (IC1,IC2)

#%%
import numpy as np
from fun import bootstrap, FT
import matplotlib.pyplot as plt
import scipy.stats as st

#Determinación de la varianza.

m=1000
sigma=1.4
media=3
muestra=FT(np.random.random(1000),0,0,1)
boot_std=bootstrap(muestra,np.std)
std_ft=boot_std.mean()
print('El valor empírico de la desvoación estándar es:',std_ft)
print('El valor teŕico de la desvoación estándar es:',np.pi/np.sqrt(6))
print('El valor estimado de pi:',std_ft*np.sqrt(6))


x=np.linspace(1,1.6,100)
y=st.norm.pdf(x,loc=boot_std.mean(),scale=boot_std.std())
plt.hist(boot_std,density=True)
#plt.hlines(st.norm.pdf(x=boot_std.mean()-boot_std.std()/2,loc=boot_std.mean(),scale=boot_std.std()),boot_std.mean()-boot_std.std()/2,boot_std.mean()+boot_std.std()/2,label='$\sigma$',linestyles='dashed',color='b')
plt.plot(x,y,label='Gaussiana',color='hotpink')
plt.vlines(boot_std.mean(),0,st.norm.pdf(boot_std.mean(),loc=boot_std.mean(),scale=boot_std.std()),colors='mediumvioletred',linestyles='dashed',label='$\sigma$')
plt.xlabel('$\sigma$')
plt.ylabel('f')
plt.title('Distribución gaussiana de la desviación estándar')

#Determinación de los intervalos de confianza.
ic=IC(muestra,np.std)
print('El intervalo de confianza al 95% es:',ic)
plt.vlines(ic[0],0,st.norm.pdf(ic[0],loc=boot_std.mean(),scale=boot_std.std()),colors='mediumvioletred',label='Intervalos de confianza')
plt.vlines(ic[1],0,st.norm.pdf(ic[1],loc=boot_std.mean(),scale=boot_std.std()),colors='mediumvioletred')
plt.legend()
 #%%
import numpy as np
from fun import bootstrap
from fun import FT
#Análisis de la varianza y ancho del intervalo de confianza en función del número de experimentos.
x=np.round(np.linspace(1000,200000,64)).astype(int)
rango=[]
for i in range(64):
    rango.append(IC(FT(np.random.random(x[i]),0,0,1),np.std)[1]-IC(FT(np.random.random(x[i]),0,0,1),np.std)[0])
plt.plot(x,rango,label='Tamaño del intervalo de confianza',linewidth=2.5)
plt.legend()
plt.xlabel('n')
plt.ylabel('$\sigma_{0.95}-\sigma_{0.05}$')
plt.show()

sigma=[]
for i in range(64):
    sigma.append(bootstrap(FT(np.random.random(x[i]),0,0,1),np.std).mean())
plt.plot(x,sigma,linewidth=2.5)
plt.legend()
plt.xlabel('n')
plt.ylabel('$\sigma$')
plt.show()

