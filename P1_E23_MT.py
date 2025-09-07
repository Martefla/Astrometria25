#Practico 1, Ejerecicio 23
#%%
#Inciso c: 

from fun import lfg
import matplotlib.pyplot as plt
m=10000
valor=[n for n in lfg(m,x0=150)]
dado=[]
for n in valor:
    if n <1/36:
        dado.append(2)
    elif n>=1/36 and n<1/12:
        dado.append(3)
    elif n>=1/12 and n<1/6:
        dado.append(4)
    elif n>=1/6 and n<10/36:
        dado.append(5)
    elif n>=10/36 and n<15/36:
        dado.append(6)
    elif n>=15/36 and n<21/36:
        dado.append(7)
    elif n>=21/36 and n<26/36:
        dado.append(8)
    elif n>=26/36 and n<30/36:
        dado.append(9)
    elif n>=30/36 and n<33/36:
        dado.append(10)
    elif n>=33/36 and n<35/36:
        dado.append(11)
    else:
        dado.append(12)



print(dado)

plt.hist(dado,11,)
plt.xlim(2,13)
plt.xlabel('Suma')
plt.plot





#%%
#Inciso d

from fun import lfg
import matplotlib.pyplot as plt
m=10000
valor1=[n*6 for n in lfg(m,x0=150)]
valor2=[n*6 for n in lfg(m,x0=51)]
dado1=[]
dado2=[]
for n in valor1:
    if n <1:
        dado1.append(1)
    elif n>=1 and n<2:
        dado1.append(2)
    elif n>=2 and n<3:
        dado1.append(3)
    elif n>=3 and n<4:
        dado1.append(4)
    elif n>=4 and n<5:
        dado1.append(5)
    else:
        dado1.append(6)

for n in valor2:
    if n <1:
        dado2.append(1)
    elif n>=1 and n<2:
        dado2.append(2)
    elif n>=2 and n<3:
        dado2.append(3)
    elif n>=3 and n<4:
        dado2.append(4)
    elif n>=4 and n<5:
        dado2.append(5)
    else:
        dado2.append(6)

suma=[]
for i in range(m-2):
    x=dado1[i]
    y=dado2[i]
    sum=x+y
    suma.append(sum)


print(suma)

plt.hist(suma,11,)
plt.xlim(2,13)
plt.xlabel('Suma')
plt.plot
