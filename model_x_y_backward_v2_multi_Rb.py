# from cProfile import label
import numpy as np
# from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
import copy
#################VARIABLES###############

pars = {'Smax':10,
                'Rs':1,
                'lambdas':0.1,
                'taux':14,
                'P': 10,
                'Rb':1.04,
                'lambdab': 0.05,
                'L':.8,
                'tauy':14,
                'S':10,
                'alpha':0.5,
                'beta':0.5,
                'tauz':1,
                'lambdaf': 1,
                'tauf': 720,
                'f': 1}

def ksi(t):
    ksi_value = np.random.normal(0,0.01)
    if ksi_value > 1:
        return 1
    if ksi_value < -1:
        return -1
    return ksi_value

def F(X): #X = [x,y,z,f]
    dx = 1/pars['taux']*(pars['Smax']/(1 + np.exp((pars['Rs']-X[1])/pars['lambdas'])) - X[0])
    dy = 1/pars['tauy']*(pars['P']/(1 + np.exp((pars['Rb'] - X[1])/pars['lambdab'])) + X[3]*pars['L'] - X[0]*X[1] + X[2])

    dz = 1/pars['tauz']*(pars['S']*(pars['alpha']*X[0] + pars['beta']*X[1])*0 - X[2])
    # df = 1/pars['tauf']*(X[1] - pars['lambdaf']*X[3])
    df = 0
    return np.array([dx, dy, dz, df])


def euler_modifie(F,X0,t,dt, direction=1):
    Y = np.zeros((len(t),4))
    Y[0,:] = X0
    X = copy.deepcopy(X0)
    start = time.perf_counter()
    for i in range(1,len(t)):
        if direction==1:
            X = X + F(X)*dt
        else:
            X = X - F(X)*dt
        Y[i,:] = X
    return Y

Rb_values = [1.01, 1.02, 1.03, 1.04, 1.05, 1.08, 1.1]
dt = 0.1
t = np.arange(0,3000, dt)
X0 = [1.4,.75, 0, 1]

# Nullclines
y = np.arange(0, 3, .1)
null_x = pars['Smax']/(1+np.exp((pars['Rs']-y)/pars['lambdas']))
null_y = (1/y)*(pars['P']/(1+np.exp((pars['Rb'] -y)/pars['lambdab']))+pars['L'])

# plot for different Rb values and save fig
for Rb in Rb_values:

    pars.update({'Rb':Rb})

    Y = euler_modifie(F,X0,t,dt, direction=-1)
    fig, axes = plt.subplots(4, 2, sharex=True)
    gs = axes[1, 0].get_gridspec()
    for ax in axes[:, 1:].ravel():
        ax.remove()

    for (i, ax), var in zip(enumerate(axes[:,0]), ['x', 'y', 'z', 'f']):
        ax.plot(t,Y[:,i],label=var)
        ax.set_ylabel(var)

    axbig = fig.add_subplot(gs[:, 1])
    axbig.plot(Y[:,1],Y[:,0])
    axbig.plot(Y[0,1],Y[0,0], marker="x")


    axbig.plot(y, null_x, label='x_null')
    axbig.plot(y, null_y, label='y_null')
    axbig.set_xlim((0,3))
    axbig.set_ylim((0,10))

    plt.title(f'{Rb}')
    plt.savefig(f'figures/limit_cycle_{Rb}.png')
    plt.legend()
plt.show()