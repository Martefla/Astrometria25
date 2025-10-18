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

mc_mg=(0.7<tb['p_mg']) & (-80<tb['petroMag_r']) & (-80<tb['petroMag_g'])
tb_mg = tb[mc_mg]

c=300000
H0=75
tb_el=tb_el[(tb_el['z']>0.000001) & (tb_el['z']<1)]
x_el=np.log(tb_el['z'])
Mr_el=tb_el['petroMag_r']-25-5*np.log10(c*tb_el['z']/H0)
plt.scatter(x_el,Mr_el,s=1,c=rojo,alpha=0.5)
plt.scatter(0,0,s=5,color=rojo,label='Elípticas')
plt.xlim(-4.5,-0.5)
plt.ylim(-24,-18)
plt.gca().invert_yaxis()
plt.xlabel('ln(z)',fontsize=14)
plt.ylabel('$M_r$',fontsize=14)
plt.legend(fontsize=14)

x=np.log(tb_el['z'])
y=Mr_el
n=len(y)
a_el=sum((y-sum(y)/n)*x)/(sum(x**2)-sum(x*sum(x))/n)
b_el=sum(y-a_el*x)/n
x1=(-4.25,-0.75)
y1=(x1[0]*a_el+b_el,x1[1]*a_el+b_el)
plt.plot(x1,y1,color='red',linewidth=2)
plt.show()


tb_sp1=tb_sp[(tb_sp['z']>0.000001) & (tb_sp['z']<1)]
x_sp=np.log(tb_sp1['z'])
Mr_sp=tb_sp1['petroMag_r']-25-5*np.log10(c*tb_sp1['z']/H0)
plt.scatter(x_sp,Mr_sp,s=1,c=azul,alpha=0.5)
plt.xlim(-5.5,-1)
plt.ylim(-23,-16)
plt.gca().invert_yaxis()
plt.xlabel('ln(z)',fontsize=14)
plt.ylabel('$M_r$',fontsize=14)
plt.legend(fontsize=14)
plt.title

x=np.log(tb_sp1['z'])
y=Mr_sp
n=len(y)
a_sp=sum((y-sum(y)/n)*x)/(sum(x**2)-sum(x*sum(x))/n)
b_sp=sum(y-a_sp*x)/n
x1=(-5.25,-1.25)
y1=(x1[0]*a_sp+b_sp,x1[1]*a_sp+b_sp)
plt.plot(x1,y1,color='b',linewidth=2.5)
plt.scatter(0,0,s=5,c=azul,label='Espirales')
plt.legend(fontsize=14)
plt.show()
print('Azules', a_sp , b_sp)

tb_mg1=tb_mg[(tb_mg['z']>0.000001) & (tb_mg['z']<1)]
x_mg=np.log(tb_mg1['z'])
Mr_mg=tb_mg1['petroMag_r']-25-5*np.log10(c*tb_mg1['z']/H0)
plt.scatter(x_mg,Mr_mg,s=5,c=amarillo,alpha=0.5)
plt.xlim(-4,-1)
plt.ylim(-23,-18)
plt.gca().invert_yaxis()
plt.xlabel('ln(z)',fontsize=14)
plt.ylabel('$M_r$',fontsize=14)
plt.legend(fontsize=14)
plt.title
print('Rojas', a_el , b_el)

x=np.log(tb_mg1['z'])
y=Mr_mg
n=len(y)
a_mg=sum((y-sum(y)/n)*x)/(sum(x**2)-sum(x*sum(x))/n)
b_mg=sum(y-a_mg*x)/n
x1=(-3.75,-1.25)
y1=(x1[0]*a_mg+b_mg,x1[1]*a_mg+b_mg)
plt.plot(x1,y1,color=amarillo,linewidth=2.5)
plt.scatter(0,0,s=5,c=amarillo,label='En fusión')
plt.legend(fontsize=14)
print('fusion', a_mg , b_mg)
plt.show()

#%%
x1=(-5.25,-1)
x1=np.array(x1)
plt.scatter(x_sp,Mr_sp,s=0.5,alpha=0.5,c=azul)
plt.plot(x1,x1*a_sp+b_sp,color='b',label='Ajuste galaxias espirales')
plt.scatter(0,0,c=azul,s=5,label='Galaxias espirales')

plt.scatter(x_el,Mr_el,s=0.5,alpha=0.3,c=rojo)
plt.plot(x1,x1*a_el+b_el,color='r',label='Ajuste galaxias elípticas')
plt.scatter(0,0,c=rojo,s=5,label='Galaxias elípticas')

plt.scatter(x_mg,Mr_mg,s=1,alpha=0.8,c='gold')
plt.plot(x1,x1*a_mg+b_mg,color='gold',label='Ajuste de las fusionadas')
plt.scatter(0,0,c='gold',s=5,label='Galaxias en fusión')

plt.xlabel('$log(z)$',fontsize=13)
plt.ylabel('$M_r$',fontsize=13)
plt.legend(fontsize=10,loc='lower right')
plt.xlim(-5.5,-0.75)
plt.ylim(-24,-16)
plt.gca().invert_yaxis()