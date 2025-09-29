#%%
#Distribución de Poisson
import matplotlib.pyplot as plt
import numpy as np

#Defino la inversa de la ocumulada de Poisson
def Proc_poisson(k,y):
    F=-np.log(1-y)/k
    return F

T=180
k=1/12
n=1000
y=np.random.random(n)
plt.hist(Proc_poisson(k,y),50,range=(0,T),density=True,color='lightcoral',label='Muestra')
x=np.linspace(0,180)
plt.plot(x,k*np.e**(-k*x),label='Teórico')
plt.xlabel('t [min]')
plt.ylabel('Número de eventos')
plt.title('Procesos de Poisson')
plt.legend()

