#Práctico 1, Ejercicio 21
#Recrear el juego
#%%
from fun import lfg
import matplotlib.pyplot as plt
n=1000
ganadora=[]
num_al=lfg(n)
for g in num_al:
    if g<0.3333:
        ganadora.append(1)
    elif g>=0.3333 and g<0.66666:
        ganadora.append(2)
    else:
        ganadora.append(3)
listelec=[]
eleccion=int(input('Seleccione una semilla distinta de 5 ;) :'))
if eleccion==5:
    print('No hay premio por hacer trampa')
else:
   
    num_al=lfg(n,x0=eleccion)
    for g in num_al:
        if g<0.3333:
            listelec.append(1)
        elif g>=0.3333 and g<0.66666:
            listelec.append(2)
        else:
            listelec.append(3)

    resultado=[]
    for i in range(n):
        gi=ganadora[i]
        ei=listelec[i]

        if gi==ei:
            resultado.append('Auto')
            print('¡Se ganó un auto!')
        else:
            resultado.append('Cabra')
            print('¡¿Se ganó una cabra?!')

plt.hist(resultado,2)
plt.plot
# %%
from fun import lfg
import matplotlib.pyplot as plt
n=100
ganadora=[]
num_al=lfg(n)
for g in num_al:
    if g<0.3333:
        ganadora.append(1)
    elif g>=0.3333 and g<0.66666:
        ganadora.append(2)
    else:
        ganadora.append(3)

eleccion=int(input('Seleccione una semilla distinta de 5 ;) :'))

listelec=[]
num_al=lfg(n,x0=eleccion)
for g in num_al:
    if g<0.3333:
        listelec.append(1)
    elif g>=0.3333 and g<0.66666:
        listelec.append(2)
    else:
        listelec.append(3)

eliminada=[]
for i in range(n):
    gi=ganadora[i]
    ei=listelec[i]
    num=[1,2,3]
    
    if gi==ei:
        numpool=num[:gi]+num[gi+1:]
    else:
        if gi<ei:
            num_pool=num[:gi]+num[gi+1:ei]+num[ei+1:]
        else:
            num_pool=num[:ei]+num[ei+1:ei]+num[ei+1:]


plt.hist(resultado,2)
plt.plot

#%%
lista=['a', 'b', 'c']
lista1=lista[:1]+lista[1+1:]
print(lista1)