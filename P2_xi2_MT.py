#%%
#Prueba de xi cuadrado

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def chi2_frec(O,E):
    '''
    Cálculo de chi cuadrado para el análisis de modelos
    
    Parámetros:
    O: lista de frecuencias de la muestra.
    E: lista de frecuencias del modelo con el mismo bineado
    
    Return:
    test: valor de xi cuadrado.
    '''
    mask = E > 0
    O = np.array(O)[mask]
    E = np.array(E)[mask]
    test=np.sum(((O-E)**2)/E)
    return(test)

#Definición de la muestra:
m=100
n=10
p=0.4
muestra=np.random.binomial(n,p,m)
x=[0,1,2,3,4,5,6,7,8,9,10]
y=st.binom.pmf(x,n,p)
plt.hist(muestra,10,density=True,label='Muestra')

#Detreminación de las frecuencias de la muestra:
y_muestra=[]
for i in range(11):
    y_muestra.append(np.sum(muestra==i)/m)


plt.scatter(x,y,c='lightcoral',label='Módelo')
plt.xlabel('x')
plt.ylabel('frecuencia')
plt.legend()
plt.title('Distribucion binomial')
plt.show()

#Cálculo de xi cuadrado:
x2=chi2_frec(y_muestra,y)
print('El valor de x cuadrado es :',x2)

m=100
df = n-1
x = np.linspace(0, 25, m)
y = st.chi2.pdf(x, df)


x95=st.chi2.ppf(0.95,df=df)
print('El valor de xi cuadrado al 95% es :',x95)
plt.plot(x, y, label=f'Chi cuadrado(df={df})',linewidth=2.5)
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.title('Distribución chi cuadrado')
plt.vlines(x2, -0.005, st.chi2.pdf(x2, df), colors='k', linestyles='solid',label='x2',linewidth=2.5)
plt.vlines(x95, -0.005, st.chi2.pdf(x95, df), colors='lightcoral', linestyles='solid',label='x2 95%',linewidth=2.5)
plt.legend()
plt.show()

if x2<x95:
    print('No se rechaza la hipótesis nula.')
else:
    print('Se rechaza la hipótesis nula.')
#%%
#Calcular en valor p
def chi2_pvalue(x2,df):
    '''
    Cálculo del valor p para xi cuadrado.
    
    Parámetros: 
    x2: valor de xi cuadrado.
    df: grados de libertad.

    Return:
    p: valor p.

    '''
    p=1-st.chi2.cdf(x2,df)
    return p

#Variación de la media de la normal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

m=100
df=n-1
sigma=2.5
mu=np.linspace(0,10,100)
p=[]
for i in range(len(mu)):
    muestra=np.random.normal(mu[i],sigma,m)
    frec_muestra, x = np.histogram(muestra, bins=10, density=True, range=(0,10))
    frec_binom=st.binom.pmf(x[:-1],10,0.4)
    x2=(chi2_frec(frec_muestra,frec_binom))
    plt.scatter(mu[i],x2,color='lightcoral')
    p.append(chi2_pvalue(x2,df))

x95=st.chi2.ppf(0.95,df=df)
plt.hlines(x95, 0, 10, linestyles='solid',label='x2 95%')
plt.xlabel('mu')
plt.ylabel('chi al cuadrado')
plt.legend()
plt.show()

plt.plot(mu,p, c='lightcoral',label='valor p al 95%')
plt.hlines(0.05, 0, 10)
plt.xlabel('mu')
plt.ylabel('valor p')
plt.legend()
plt.show()
#%%
#Cambio de n y la media de la normal
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

m=10000
df=10-1
sigma=15

mu=np.linspace(350,450,100)
p=[]
binom=np.random.binomial(1000,0.4,m)
frec_binom, x = np.histogram(binom, bins=100, density=True)

for i in range(len(mu)):
    muestra=np.random.normal(mu[i],sigma,m)
    frec_muestra, x = np.histogram(muestra, bins=100, density=True, range=(x[0],x[-1]))
    x2=(chi2_frec(frec_muestra,frec_binom))
    plt.scatter(mu[i],x2,color='lightcoral')
    p.append(chi2_pvalue(x2,df))

x95=st.chi2.ppf(0.95,df=df)
plt.hlines(x95, 350, 450, linestyles='solid',label='x2 95%')
plt.xlabel('mu')
plt.legend()
plt.ylabel('chi al cuadrado')
plt.show()

plt.plot(mu,p, c='lightcoral')
plt.hlines(0.05, 350, 450,label='valor p al 95%')
plt.xlabel('mu')
plt.legend()
plt.ylabel('valor p')
plt.show()

