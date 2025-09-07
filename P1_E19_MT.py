#P1_E19_MT

#%%
#Generador de Fibonacci con retardo
from fun import glc
import matplotlib.pyplot as plt
import numpy as np

def lfg(n,m=2**32,k=55,j=24,x0=5):
    num_al=[]
    num_al1=[]
    if k>j:
        num_al=[g for g in glc(k,16807,0,2**31-1,x0)]
    else:
        num_al=[g for g in glc(j,16807,0,2**31-1,x0)]

    for i in range(n):
        xk=num_al[-k]
        xj=num_al[-j]
        x=(xk+xj) % m

        #Agregar un objeto a la lista
        num_al.append(x)

    num_al1=[n/m for n in num_al]
        
    return num_al1



num_al1=lfg(10000)
av=np.average(num_al1)
var=np.var(num_al1)
print('El promedio es:',av)
print('La varianza es:',var)

plt.hist(num_al1)
plt.xlabel('$x_n$')


# %%
#Generador de números random con numpy.
import numpy as np

num_al=np.random.random(10000)

av=np.average(num_al)
var=np.var(num_al)
print('El promedio es:',av)
print('La varianza es:',var)

plt.hist(num_al)
plt.xlabel('$x_n$')

# %%
