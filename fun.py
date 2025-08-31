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