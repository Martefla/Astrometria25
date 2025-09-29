#Transformada inversa Fisher-Tippett
#%%
import matplotlib.pyplot as plt
import numpy as np

#Defino la función inversa de la acumulada de Fisher Tippett como:

def FT(y,mu,xi,lamb):
    if xi==0:
        f=-np.log(-np.log(y))/lamb+mu
    else:
        f=((-np.log(y))**(-xi)-1)/(xi*lamb)+mu
    return f

#Creo la lista de números aleatorios de distrubución constante entre 0 y 1:
n=10000
y=np.random.random(n)
mu=2
xi=0.5
lamb=1.25
x=FT(y,mu,xi,lamb)
plt.hist(x,100,label='xi=/=0',density=True,range=(-3,15))
x=np.linspace(0,15,100)
plt.plot(x,lamb*((1+xi*lamb*(x-mu))**(-1/xi))**(xi+1)*np.e**(-((1+xi*lamb*(x-mu))**(-1/xi))),label='Modelo',color='darkblue')


np.log
mu=0
xi=0
lamb=1
x=FT(y,mu,xi,lamb)
plt.hist(x,100,label='xi=0', color='r',alpha=0.5,density=True)
x=np.linspace(-3,10,100)
plt.plot(x,lamb*np.e**(-(lamb*(x-mu)*(xi+1)+np.e**(-lamb*(x-mu)))),label='Modelo',color='purple')

plt.legend()
plt.xlabel('x')
plt.ylabel('f')
plt.title('Histograma muestras Fisher-tippett')

#%%
from fun import FT
import numpy as np
import matplotlib.pyplot as plt
#Comparación del valor toerico del valor de espectación
# de una función de Fisher-Tippet de tipo 1 con el calculado.
n=100000
E=[]
x=[]
teoria=[]
m=100
for i in range(m):
    x_i=i*0.5+5
    x.append(x_i)
    teoria.append(0.57721/x_i)
    y=np.random.random(n)
    E.append(np.mean(FT(y,0,0,x_i)))

plt.plot(x,E,label='Empírico',linewidth=4)
plt.plot(x,teoria,label='Teórico',color='r',alpha=0.5,linewidth=4)
plt.xlabel('$\lambda$')
plt.ylabel('<x>')
plt.legend()
plt.title('Comparación valor de expectación')

#t(x)=((1+xi**(lamb*(x-mu)))**(-1/xi))
#f(x)=lamb*t(x)**(xi+1)*np.e**(-t(x))

#f(x)=lamb*((1+xi**(lamb*(x-mu)))**(-1/xi))**(xi+1)*np.e**(-((1+xi**(lamb*(x-mu)))**(-1/xi)))
