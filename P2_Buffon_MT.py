#%%
#BUFON 
import numpy as np
import matplotlib.pyplot as plt

#Planteo las variables aleatorias.
n=100000000
l=1
t=2
resultado=[]
tita=np.random.random(n)*np.pi
x=np.random.random(n)*t/2

#Determino si hubo intersección o no.
for i in range(n):
    x_c=np.sin(tita[i])*l/2
    if x[i]<=x_c:
        resultado.append('Tocó')
    else:
        resultado.append('No tocó')
P=resultado.count('Tocó')/n

#Calculo de la probabilidad y de pi.
pi=2*l/(P*t)
print('Porcentaje de intesecciones:',P*100,'%')
print('Aproximación de pi:', pi)