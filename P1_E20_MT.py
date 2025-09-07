# Practico 1, Ejercicio 20
#%%
#Cálculo del parámetro de Pearson.
import numpy as np
def pearson_correlation (x , y ) :
 """
 Calcula el coeficiente de correlacion de Pearson entre dos arrays

 Parameters :
 x, y: arrays de igual longitud

 Returns :

 r: coeficiente de correlacion de Pearson

 """
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
from fun import pearson_correlation, glc, lfg

ret=[1,2,3,5,7,10]
n=1000
glc256=glc(n)
glc31=glc(n,16807,0,2**31-1)
lfg=lfg(n)

pearson=[]
for i in ret:
  x=glc256[:-i]
  y=glc256[i:]
  pearson.append(pearson_correlation(x,y))

print(pearson)

pearson=[]
for i in ret:
  x=glc31[:-i]
  y=glc31[i:]
  pearson.append(pearson_correlation(x,y))

print(pearson)

pearson=[]
for i in ret:
  x=lfg[:-i]
  y=lfg[i:]
  pearson.append(pearson_correlation(x,y))

print(pearson)

