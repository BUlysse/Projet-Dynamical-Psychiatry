from cProfile import label
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#################VARIABLES###############

tx = 14
Smax = 10
Rs = 1
lambs = 0.1

ty = 14
P = 10
Rb = 1.04
lambb = 0.05
L = 1.01

tz = 1
S = 10
alpha = 0.5
beta = 0.5

tf = 720
lambf = 1

def ksi(t):
    ksi_value = np.random.normal(0,0.01)
    if ksi_value > 1:
        return 1
    if ksi_value < -1:
        return -1
    return ksi_value

X0 = [0.05,0.05,0,0]

#################EQUATIONS################

def euler_modifie(F,X0,t):
    Y = np.array([X0])
    for i in range(len(t)-1):
        k1 = F(Y[i,:],t[i])
        k2 = F(Y[i,:] + k1*(t[i+1]-t[i]),t[i+1])
        Y=np.append(Y,[Y[i,:] + (k1+k2)*(t[i+1]-t[i])/2], axis = 0)
    return Y

def F(X,t): #X = [x,y,z,f]
    dx = 1/tx*(Smax/(1 + np.exp((Rs-X[1])/lambs)) - X[0])
    dy = 1/ty*(P/(1 + np.exp((Rb - X[1])/lambb)) + X[3]*L - X[0]*X[1] + X[2])
    dz = 1/tz*(S*(alpha*X[0] + beta*X[1])*ksi(t) - X[2])
    df = 1/tf*(X[1] - lambf*X[3])
    return np.array([dx, dy, dz, df])

################CODE#######################

t = np.arange(0,8000,0.01)

Y = euler_modifie(F,X0,t)
#Y = odeint(F,X0,t)

plt.subplot(2,1,1)
plt.plot(t,Y[:,0],label='x')
plt.plot(t,Y[:,1],label='y')
plt.plot(t,Y[:,2],label='z')
plt.plot(t,Y[:,3],label='f')

plt.legend()

plt.subplot(2,1,2)
plt.plot(Y[:,0],Y[:,1])
plt.title("diagramme de phase de x en fonction de y")

plt.show()