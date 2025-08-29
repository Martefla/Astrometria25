#%%
#EJERCICiO 18
#definimos, el generador lineal de congruencia.
#Se toma una semilla (x0) con un incremento (c) para armar una función lineal, a esta se le toma modulo (M).

#Para este ejercicio usamos (a, c, M, x0) = (57, 1, 256, 10).

def glc1(a,c,M,x0):
    x=(a*x0 + c) % M
    return x
# %%
#Se usa el número anterior para generar el siguiente.
#Usamos listas, estas pueden contener diferentes tipos de objetos, vamos a usar números asi que es como un vector.


#Para buscar buena combinacion de números, buscar en internet.
def glc(n,a=57,c=1,M=256,x0=10):

    #crea una lista vacia
    num_al=[]

    for i in range(n):
        x=(c+a*x0)%M

        #Agregar un objeto a la lista
        num_al.append(x)
        x0=x

        
    return num_al

# %%
#Busquemos el periodo. Una vez que se repite algun numero de la secuencia.
#Solo puede generar un numero maximo de M-1 numeros, si la eleccion de parametros es mala v a ser menor.
semilla=10
num_al=glc(257,x0=semilla, a=37,M=156)

for i, n in enumerate(num_al):
    if n==semilla:
        print(i+1)
        #i+1 porque python empieza de 0
        #break para parar el loop
        break
        


