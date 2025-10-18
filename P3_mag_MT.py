#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rojo='tomato'
azul='royalblue'
amarillo='goldenrod'

tb=pd.read_csv('Tb_AM_P3_Martina_Tetzlaff.csv')

mc_sp=(0.7<tb['p_cs']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_sp = tb[mc_sp]

mc_el=(0.7<tb['p_el']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_el = tb[mc_el]


x_sp=tb_sp['petroMag_g']
y_sp=tb_sp['petroMag_r']
plt.scatter(tb['petroMag_g'][mc_sp],tb['petroMag_r'][mc_sp],s=1,alpha=0.5,c=azul)
plt.xlim(10,22)
plt.ylim(10,22)
plt.scatter(0,0,s=5,label='Galaxias espirales',c=azul)
plt.xlabel('$M_g$',fontsize=13)
plt.ylabel('$M_r$',fontsize=13)
plt.legend(fontsize=14)

n=len(y_sp)
a_sp=sum((y_sp-sum(y_sp)/n)*x_sp)/(sum(x_sp**2)-sum(x_sp*sum(x_sp))/n)
b_sp=sum(y_sp-a_sp*x_sp)/n
x1=(11,21)
y1=(x1[0]*a_sp+b_sp,x1[1]*a_sp+b_sp)
plt.plot(x1,y1,color='b')
plt.show()
print('Parámetros del ajuste lineal para espirales: a=', a_sp ,'b=', b_sp)


mc=(-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb=tb[mc]
x_el=tb_el['petroMag_g']
y_el=tb_el['petroMag_r']
plt.scatter(x_el,y_el,s=1,alpha=0.5,c=rojo)
plt.xlim(12,25)
plt.ylim(12,21)
plt.scatter(0,0,c=rojo,s=5,label='Galaxias elípticas')
plt.xlabel('$M_g$',fontsize=13)
plt.ylabel('$M_r$',fontsize=13)
plt.legend(fontsize=14)

n=len(y_el)
a_el=sum((y_el-sum(y_el)/n)*x_el)/(sum(x_el**2)-sum(x_el*sum(x_el))/n)
b_el=sum(y_el-a_el*x_el)/n
x1=(12.5,22.5)
y1=(x1[0]*a_el+b_el,x1[1]*a_el+b_el)
plt.plot(x1,y1,color='red')
plt.show()
print('Parámetros del ajuste lineal para elípticas: a=', a_el ,'b=', b_el)


mc=(-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb=tb[mc]
tb_mg=tb[(0.7<tb['p_mg'])]
x=tb_mg['petroMag_g']
y=tb_mg['petroMag_r']
plt.scatter(x,y,s=1,alpha=0.5,c=amarillo)
plt.xlim(14,20)
plt.ylim(13,19)
plt.scatter(0,0,c=amarillo,s=5,label='Galaxias en fusión')
plt.xlabel('$M_g$',fontsize=13)
plt.ylabel('$M_r$',fontsize=13)
plt.legend(fontsize=14)

n=len(y)
a_mg=sum((y-sum(y)/n)*x)/(sum(x**2)-sum(x*sum(x))/n)
b_mg=sum(y-a_mg*x)/n
x1=(14.5,19.5)
y1=(x1[0]*a_mg+b_mg,x1[1]*a_mg+b_mg)
plt.plot(x1,y1,color=amarillo)
plt.show()
print('Parámetros del ajuste lineal para galaxias en fusión: a=', a_mg ,'b=', b_mg)


x1=(13,21)
x1=np.array(x1)
plt.scatter(x_sp,y_sp,s=1,alpha=0.5,c=azul)
plt.plot(x1,x1*a_sp+b_sp,color='b',label='Ajuste galaxias espirales')
plt.scatter(0,0,c=azul,s=5,label='Galaxias espirales')

plt.scatter(x_el,y_el,s=1,alpha=0.3,c=rojo)
plt.plot(x1,x1*a_el+b_el,color='r',label='Ajuste galaxias elípticas')
plt.scatter(0,0,c=rojo,s=5,label='Galaxias elípticas')

plt.scatter(x,y,s=1,alpha=0.8,c='gold')
plt.plot(x1,x1*a_mg+b_mg,color='gold',label='Ajuste de las fusionadas')
plt.scatter(0,0,c='gold',s=5,label='Galaxias en fusión')

plt.xlabel('$M_g$',fontsize=13)
plt.ylabel('$M_r$',fontsize=13)
plt.legend(fontsize=11)
plt.xlim(12,22)
plt.ylim(11,21)
