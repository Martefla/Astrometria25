#%%
#Analisis de los colores de la muestra de galaxias.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rojo='tomato'
azul='royalblue'
amarillo='goldenrod'

tb=pd.read_csv('Tb_AM_P3_Martina_Tetzlaff.csv')

mc_sp=(0.7<tb['p_cs']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_sp = pd.read_csv('Tb_AM_P3_Martina_Tetzlaff.csv')[mc_sp]

SPu_g=tb_sp['petroMag_u'][mc_sp]-tb_sp['petroMag_g'][mc_sp]
SPg_r=tb_sp['petroMag_g'][mc_sp]-tb_sp['petroMag_r'][mc_sp]

mc_el=(0.7<tb['p_el']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_el = pd.read_csv('Tb_AM_P3_Martina_Tetzlaff.csv')[mc_el]
ELu_g=tb_el['petroMag_u'][mc_el]-tb_el['petroMag_g'][mc_el]
ELg_r=tb_el['petroMag_g'][mc_el]-tb_el['petroMag_r'][mc_el]

plt.hist(ELu_g,label='Elípticas',color=rojo,range=(-1,5),bins=50)
plt.hist(SPu_g,label='Espirales',alpha=0.5,range=(-1,5),bins=50,color=azul)
plt.xlabel('u-g',fontsize=13)
plt.legend(fontsize=13)
plt.title('Histogramas colores u-g',fontsize=15)
plt.show()

plt.hist(ELg_r,label='Elípticas',color=rojo,range=(-0,3),bins=50)
plt.hist(SPg_r,label='Espirales',color=azul,alpha=0.5,range=(0,3),bins=50)
plt.xlabel('g-r',fontsize=13)
plt.legend(fontsize=13)
plt.title('Histogramas colores g-r',fontsize=15)
plt.show()
#%%
#Comparación estadística de las distribuciones de colores u-g y g-r entre galaxias elípticas y espirales.
from fun import x2_valorp_frec

bin=10
frec_elug, x = np.histogram(ELu_g, bins=bin, density=True, range=(-1,5))
frec_spug, x = np.histogram(SPu_g, bins=bin, density=True, range=(-1,5))

df=bin-1
(x2,p)=x2_valorp_frec(frec_elug,frec_spug,df)
print('Valor p para el color u-g:',p)

if p<0.05:
    print(' * p<0.05 => Las distribuciones no son consistentes entre sí')
else:
    print(' * p>0.05 => Las distribuciones son consistentes entre sí')


frec_elgr, x = np.histogram(ELg_r, bins=bin, density=True, range=(0,3))
frec_spgr, x = np.histogram(SPg_r, bins=bin, density=True, range=(0,3))

df=bin-1
(x2,p)=x2_valorp_frec(frec_elgr,frec_spgr,df)
print('Valor p para el color g-r:',p)

if p<0.05:
    print(' * p<0.05 => Las distribuciones no son consistentes entre sí')
else:
    print(' * p>0.05 => Las distribuciones son consistentes entre sí')
