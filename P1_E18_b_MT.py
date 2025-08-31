#
#Practico 1, Ejercicio 18, Inciso b
#%%
#Generar pasos aleatorios
import math
px=[]
py=[]
xi=0
yi=0
for i in range(10):
    n=100
    m=2**31-1
    x0=i*13051
    deltax=[]
    num_al=[]
    num_al=glc(n,16807, 0, m, x0)
    deltax=[y*math.sqrt(2)*2/m-math.sqrt(2) for y in num_al]

    deltay=[]
    num_al=[]
    num_al=glc(n,16807, 0, m, -x0*2+3)
    deltay=[y*math.sqrt(2)*2/m-math.sqrt(2) for y in num_al]

    for x in deltax:
        xi=xi+x
    for y in deltay:
        yi=yi+y
    px.append(xi)
    py.append(yi)

print(px)
print(py)

#Graficar las posiciones finales
x=px
y=py

plt.plot(x,y,'ro',label='pares')
plt.xlabel("x")
plt.ylabel('y')
plt.title('Posición')
plt.legend('')
# %%


