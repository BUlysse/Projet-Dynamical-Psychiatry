import matplotlib.pyplot as plt
import numpy as np
from random import random
from random import seed
import matplotlib.pyplot as plt
import time

plt.rcParams.update({'font.size': 28})


#Bipolar cycle rapide:
Par={'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':1.0, 'lambdab': 0.05, 'L':0.6, 'tauy':14, 'S':4.5, 'alpha':0.5, 'beta':0.5, 'tauz':1 }



Li=[o for o in range(800000)]
dt=0.01
Lt=[dt*i for i in Li]
y=0.1
x=0.0
z=0.0
f=0.0
Ly=[]
Lx=[]
Lz=[]
Lf=[]
seed(10)


Smax = Par['Smax']
Rs = Par['Rs']
lambdas = Par['lambdas']
taux = Par['taux']
P = Par['P']
Rb = Par['Rb']
lambdab = Par['lambdab']
L = Par['L']
tauy = Par['tauy']
S = Par['S']
alpha = Par['alpha']
beta = Par['beta']
tauz = Par['tauz']

L = 0.2
Rb = 1.04
S = 4



start = time.perf_counter()
for i in Li:

    #b=random()
    a=(random()-0.5)
   # La.append(3.4+a)
    #print(a)
    dx=(Smax/(1+np.exp((Rs-y)/lambdas))-x)/taux
    dy=(P/(1+np.exp((Rb-y)/lambdab))+L*f-y*x-z)/tauy
    dz=(S*(alpha*x+beta*y)*(a)-z)/tauz
    df=(y-1.0*f)/720
    y=y+dy*dt#+a
    x=x+dx*dt
    z=z+dz*dt
    f=f+df*dt

   # changes in parameters during simulation
   # pharmacological intervention
    if i==547500:
      z=-5

    if i==547501:
      print([dx, dy, dz, df])



    Ly.append(y)
    Lx.append(x)
    Lz.append(z)
    Lf.append(f)

print(time.perf_counter() - start)

cutT=0
fig, axs = plt.subplots(4)
axs[0].plot(Lt[cutT::],Lx[cutT::], 'b')
axs[0].axes.xaxis.set_ticklabels([])
#axs[0].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[0].set_ylabel(r'$\bf{x}$') #+'\n"symptom rate"')
axs[1].plot(Lt[cutT::],Ly[cutT::], 'magenta')
axs[1].axes.xaxis.set_ticklabels([])
axs[1].text(5400, 1.4, r'$\bf{|}$', transform=axs[1].transData)
axs[1].text(5401, 1.4, r'$\bf{-}$', transform=axs[1].transData)
axs[1].text(5402, 1.4, r'$\bf{>}$', transform=axs[1].transData)
#axs[1].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[1].set_ylabel(r'$\bf{y}$') #+'\n"internal potentiation"')
#axs[1].text(1500, 1.2, '|'+r"intervention on $P$", transform=axs[1].transData)
axs[2].plot(Lt[cutT::],Lz[cutT::], 'r')
axs[2].set_ylabel(r'$\bf{z}$') #+'\n"perceived environment"')

#axs[2].tick_params(axis='both', left='on', top='off', right='off', bottom='off', labelleft='on', labeltop='off', labelright='off', labelbottom='off')
axs[2].set_ylim(-5, 5)
axs[2].axes.xaxis.set_ticklabels([])
axs[3].plot(Lt[cutT::],Lf[cutT::], 'g')
axs[3].set_ylabel(r'$\bf{f}$')
axs[3].set_xlabel('Time (days)')
axs[3].set_ylim(0,1.5)
fig.align_ylabels()
#plt.tight_layout()

plt.show()
