#Practico 1, Ejercicio 18, inciso a.
#~~~~~~~~~~~~~~~~~~Inciso~a~~~~~~~~~~~~~~~~~~~
#%%
#Generación de numeros aleatorios por congruencia.


def glc(n,a=57,c=1,M=256,x0=10):
    """Generador de números aleatorios por congruencia lineal
    
    Parámetros:
    n:cantidad de números a generar
    a: multiplicador
    c: incremento
    M: modulo
    x0: semilla
     Salida tipo lista.
     """

    #crea una lista vacia
    num_al=[]

    for i in range(n):
        x=(c+a*x0)%M

        #Agregar un objeto a la lista
        num_al.append(x)
        x0=x

        
    return num_al

glc_1=[x/256 for x in glc(100)]
print(glc_1)
#%%
#Determinar el periodo.


from fun import glc
x0=10
num_al=glc(1000,57,1,256,x0)

for i, n in enumerate(num_al):
    if n==x0:
        print(i+1)
        break




#%%
#Analisis de peridicidad graficamente.

import matplotlib.pyplot as plt
from fun import glc

num_al=glc(1000)
x=num_al[:-1]
y=num_al[1:]

plt.plot(x,y,'ro',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
plt.title('Periodicidad')
plt.legend('')



# %%
#Analisis de periodicidad para otros números.
import matplotlib.pyplot as plt
from fun import glc

num_al=glc(1000,16807,0,2**31-1,10)
x=num_al[:-1]
y=num_al[1:]

plt.plot(x,y,'ro',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
plt.title('Periodicidad')
plt.legend('')

# %%
