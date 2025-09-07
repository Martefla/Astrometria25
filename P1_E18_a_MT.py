#Practico 1, Ejercicio 18, inciso a.
#~~~~~~~~~~~~~~~~~~Inciso~a~~~~~~~~~~~~~~~~~~~
#%%
#Generación de numeros aleatorios por congruencia.
import matplotlib.pyplot as plt

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

x=glc_1[:-1]
y=glc_1[1:]
plt.plot(x,y,'bo')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+1}$')

#%%
#Determinar el periodo.


from fun import glc
x0=10
num_al=glc(1000,57,1,256,x0)

for i, n in enumerate(num_al):
    if n==x0:
        print('El periodo es:',i)
        break

#Analisis de periodicidad graficamente.

import matplotlib.pyplot as plt

num_al=glc(1000)
x=num_al[:-1]
y=num_al[1:]

plt.plot(x,y,'bo',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
plt.title('Periodicidad')
plt.legend('')



# %%
#Analisis de periodicidad para otros números.
import matplotlib.pyplot as plt
from fun import glc
m=100
num_al=glc(m,16807,0,2**31-1,10)
x=num_al[:-1]
y=num_al[1:]

plt.plot(x,y,'b.',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
plt.legend('')

for i, n in enumerate(num_al):
    if n==x0:
        print('El periodo es:',i)
        break

# %%
import matplotlib.pyplot as plt
from fun import glc
m=10000
num_al=glc(m,16807,0,2**31-1,10)
x=num_al[:-1]
y=num_al[1:]

plt.plot(x,y,'b.',label='pares')
plt.xlabel("$n_{i}$")
plt.ylabel('$n_{i+1}$')
plt.legend('')

for i, n in enumerate(num_al):
    if n==x0:
        print('El periodo es:',i)
        break

    

# %%
#Analisis de momentos.

def mom_k(k,x):
    """
    Calculo de momentos de una lista de números.

    Parametros;
    k: orden del momento a calcular.
    x: lista de numeros.

    Return;
    mom_k: momento resultante.

    """
    n=len(x)
    mom_k_n=0
    for i in range(n):
        xi=x[i]
        mom_k_n= mom_k_n+xi**k
    mom_k=mom_k_n/n
    return mom_k

#Comparación de momentos calculados y teoricos.
from fun import glc
from fun import mom_k
print('k    1   3   7')
n=10
fn=glc(n)
x=[f/256 for f in fn]
frac_momentos=[]
ks=[1,3,7]
for k in ks:
    xk=mom_k(k,x)*(k+1)
    frac_momentos.append(xk)

print('N=10',frac_momentos)

n=100
fn=glc(n)
x=[f/256 for f in fn]
frac_momentos=[]
ks=[1,3,7]
for k in ks:
    xk=mom_k(k,x)*(k+1)
    frac_momentos.append(xk)

print('N=100',frac_momentos)

n=1000
fn=glc(n)
x=[f/256 for f in fn]
frac_momentos=[]
ks=[1,3,7]
for k in ks:
    xk=mom_k(k,x)*(k+1)
    frac_momentos.append(xk)

print('N=1000',frac_momentos)


# %%