#
#Practico 1, Ejercicio 18, Inciso b

# %%
#Crear arrays
import numpy as np
from fun import glc
import matplotlib.pyplot as plt

n_c=10
n_p=1000
x=np.zeros((n_c,n_p))#Crea una matriz 10x1000 (filasxcolumnas) con todos ceros
y=np.zeros((n_c,n_p))#Crea una matriz 10x1000 (filasxcolumnas) con todos ceros
x0=0
for i in range(n_c):
    x0=i+13
    saltox=glc(n_p,16807,0,2**31-1,x0=x0)
    saltoy=glc(n_p,16807,0,2**31-1,x0=x0+17)
    for j in range(1,n_p):
        sx=saltox[j]/(2**31-1)*2*np.sqrt(2)-np.sqrt(2)
        x[i,j]=x[i,j-1]+sx
        sy=saltoy[j]/(2**31-1)*2*np.sqrt(2)-np.sqrt(2)
        y[i,j]=y[i,j-1]+sy
    plt.plot(x[i],y[i])#Toma la fila i como una lista, para hacerlo con un acolumna es: [:,i]
plt.xlabel('x')
plt.ylabel('y')
#%%
from fun import mom_k
expt=[]
mkx=[]
for j in range(n_c):
    for i in range(n_p-1):
        x_i=x[j,:i+1]
        mkx.append(mom_k(1,x_i))    
    plt.plot(mkx)
    mkx=[]
plt.xlabel('$n_p$')
plt.ylabel('x')

