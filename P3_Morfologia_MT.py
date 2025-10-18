#%%
#Histograma de la distribución de tipos morfológicos.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fun import x2_valorp_frec

rojo='tomato'
azul='royalblue'
amarillo='goldenrod'

tb=pd.read_csv('Tb_AM_P3_Martina_Tetzlaff.csv')

mc_sp=(0.7<tb['p_cs']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_sp = tb[mc_sp]

mc_el=(0.7<tb['p_el']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_el = tb[mc_el]

mc_mg=(0.7<tb['p_mg']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_mg=tb[mc_mg]
x=('Elípticas','Espirales','En fusión')
y=(len(tb_el['z']),len(tb_sp['z']),len(tb_mg['z']))
colores=(rojo,azul,amarillo)
plt.bar(x,y,color=colores)
f_m=sum(y)/len(y)
plt.hlines(f_m,-0.4,2.4,colors='k',linestyles='dashed',alpha=0.5,linewidth=3,label='Valor medio de galaxias')

(x2,p)=x2_valorp_frec(y,(f_m,f_m,f_m),3)
print('Valor de p entre la distribución de morfologias y la uniforme:',p)
plt.ylabel('Número de galaxias',fontsize=13)
plt.legend(fontsize=12)
if p<0.05:
    print(' * p<0.05 => Las distribuciones no son consistentes entre sí')
else:
    print(' * p>0.05 => Las distribuciones son consistentes entre sí')

