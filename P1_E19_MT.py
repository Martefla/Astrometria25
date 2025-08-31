P1_E19_MT

#%%
#Generador de Fibonacci con retardo
from fun import glc
import matplotlib.pyplot as plt

def lfg(n,m=2**32,k=55,j=24,x0=5):

    if k>j:
        num_al=[g/(2**31-1) for g in glc(k,16807,0,2**31-1,x0)]
    else:
        num_al=[g/(2**31-1) for g in glc(j,16807,0,2**31-1,x0)]

    for i in range(n):
        xk=num_al[-k]
        xj=num_al[-j]
        x=((xk+xj) % m)/m

        #Agregar un objeto a la lista
        num_al.append(x)


        
    return num_al

num_al=lfg(10000)
x=num_al[56:-1]

y=num_al[57:]

#Argumentos: eje de las abcisas, eje de las ordenadas, color y tipo de punto, label
plt.plot(x,y,'ro',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
#plt.xscale('log')
#plt.yscale('log')
plt.title('Periodo')
plt.legend('')

# %%
#Generador de números random con numpy.
import numpy as np

num_al=np.random.random(1000)

x=num_al[:-1]

y=num_al[1:]

#Argumentos: eje de las abcisas, eje de las ordenadas, color y tipo de punto, label
plt.plot(x,y,'ro',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
#plt.xscale('log')
#plt.yscale('log')
plt.title('Periodo')
plt.legend('')
# %%
