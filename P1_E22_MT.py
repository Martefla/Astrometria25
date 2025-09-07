#P1_E22_MT
#%%
#Generador de tipos de galaxias.
from fun import lfg

def tipo_galaxias(n):
    galaxias=[]

    x=lfg(n)
    for x in x:
        if x<0.4:
            galaxias.append('eliptica')
        elif x>=0.4 and x<0.7:
            galaxias.append('espiral')
        elif x>=0.7 and x<0.9:
            galaxias.append('enana')
        else:
            galaxias.append('irregular')
    return galaxias

n=1000000
galaxias=tipo_galaxias(n)
print('*Galaxias elipticas:',galaxias.count('eliptica'))
print(' Porcentaje:',galaxias.count('eliptica')*100/n)
print('*Galaxias espirales:',galaxias.count('espiral'))
print(' Porcentaje:',galaxias.count('espiral')*100/n)
print('*Galaxias enanas:',galaxias.count('enana'))
print(' Porcentaje:',galaxias.count('enana')*100/n)
print('*Galaxias irregulares:',galaxias.count('irregular'))
print(' Porcentaje:',galaxias.count('irregular')*100/n)

# %%
