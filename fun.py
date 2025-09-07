#Funciones

#%%
#Generador de números aleatorios
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


def pearson_correlation (x , y ) :
 """
 Calcula el coeficiente de correlacion de Pearson entre dos arrays

 Parameters :
 x, y: arrays de igual longitud

 Returns :

 r: coeficiente de correlacion de Pearson

 """
 import numpy as np
 # Verificar que tienen la misma longitud
 if len( x ) != len( y ) :
    raise ValueError ("Los arrays deben tener la misma longitud ")

 n = len( x )

 # Calcular medias
 mean_x = np . mean ( x )
 mean_y = np . mean ( y )

 # Calcular numerador y denominador
 numerator = np .sum (( x - mean_x ) * ( y - mean_y ) )
 denominator = np . sqrt ( np .sum (( x - mean_x ) **2) * np .sum (( y - mean_y ) **2) )

 # Evitar division por cero
 if denominator == 0:
    return 0
 return numerator / denominator 

#%%
#Analisis de momentos

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

#%%
#Generador de Fibonacci con retardo
from fun import glc
import matplotlib.pyplot as plt

def lfg(n,m=2**32,k=55,j=24,x0=5):

    """
    Generador de números aleatorios de Fibonacci con Retardo.
    
    Parametros:
    n: cantidad de números a generar.
    m=2^32: módulo.
    k: primer retardo.
    j: segundo retardo.
    x0: semilla
    
    Returns:
    num_al1: lista de numeros aleatorios entre el 0 y el 1."""
    num_al=[]
    if k>j:
        num_al=[g for g in glc(k,16807,0,2**31-1,x0)]
    else:
        num_al=[g for g in glc(j,16807,0,2**31-1,x0)]
    
    
    for i in range(n):
        xk=num_al[-k]
        xj=num_al[-j]
        x=((xk+xj) % m)

        #Agregar un objeto a la lista
        num_al.append(x)
    if k>j:
        num_al1=[n/m for n in num_al[k:]]
    else:
        num_al1=[n/m for n in num_al[j:]]
    
        
    return num_al1

