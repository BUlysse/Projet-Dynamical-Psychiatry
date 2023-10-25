import PyDSTool as pdst
import numpy as np
from PyDSTool.Toolbox.phaseplane import *
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

def funtion_model (t, X, Rb, Smax=10, Rs=1,lambdas=0.1,taux=14,P= 10, lambdab= 0.05,L=0.8,tauy=14,S=10,alpha=0.5,beta=0.5,tauz=1,lambdaf= 1,tauf= 720,a= 0,f= 1):
    x= X[0]
    y = X[1]
    z = X[2]
    xstr = (Smax/(1+np.exp((Rs-y)/lambdas))-x)/taux
    ystr = (P/(1+np.exp((Rb-y)/lambdab))+L*f-y*x-z)/tauy
    zstr = (S*(alpha*x+beta*y)*0-z)/tauz
    return [xstr, ystr, zstr]


## Plot example of finding a limit cycle
# time parameter [0, 300] can be changed to [0, -300] to solve backward thus invert stability 
sol1 = solve_ivp(funtion_model, [0, 300], [1.409, 2.15, 0.], args=(1.04,), max_step=.1)
plt.plot(np.array(sol1.y).swapaxes(0, 1))

plt.plot(sol1.y[0,:])
plt.plot(np.gradient(sol1.y[0,:]))
plt.plot((np.abs(np.gradient(sol1.y[0,:]))<5e-4)/100)
plt.show()

# identify time points of one period on the previous plot
period_start = 905
period_end = 1314

circle = np.concatenate([np.expand_dims(np.array(sol1.t[period_start:period_end]), 0),
                         np.array(sol1.y)[:, period_start:period_end]])

plt.figure()
plt.plot(circle[1], circle[2])
plt.show()

# PyDSTool/Toolbox/adjointPRC.py
def makeModel(dt, params_value, ic_args):
    '''
    dt : simulations timestep
    params_value : parameters values
    ic_args : initial conditions for the variables
    '''
    #definition of model's equations
    xstr = '(Smax/(1+exp((Rs-y)/lambdas))-x)/taux'
    ystr = '(P/(1+exp((Rb-y)/lambdab))+L*f-y*x-z)/tauy'
    zstr = '(S*(alpha*x+beta*y)*0-z)/tauz'
    # fstr = '(y-lambdaf*f)/tauf'
    # fstr = '0'
    #simulations parameters
    DSargs = pdst.args()
    DSargs.varspecs = {'x': xstr, 'y': ystr, 'z': zstr}# 'f': fstr}
    DSargs.xdomain = {'x': [0, 10], 'y': [0, 4], 'z': [-5, 5]}# 'f': [.19,.21]} #variables bounds
    DSargs.pars = params_value
    DSargs.algparams = {'init_step':dt, 'max_step': dt*1.5,'max_pts': 300000, 'refine': 1}
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = 'Dynamical_Psychiatry'
    DSargs.tdomain = [0, 3000]
    return pdst.Generator.Vode_ODEsystem(DSargs)


# ------------------------------------------------------------
params_value = {'Smax':10,
                'Rs':1,
                'lambdas':0.1,
                'taux':14,
                'P': 10,
                'Rb':1.08,
                'lambdab': 0.05,
                'L':0.8,
                'tauy':14,
                'S':10,
                'alpha':0.5,
                'beta':0.5,
                'tauz':1,
                'lambdaf': 1,
                'tauf': 720,
                'a': 0,
                'f': 1}

ic_args = {'x': .1, 'y': 0.6, 'z': 0}

# List of Rb values to explore
Rb_values = [1.03]

# colors corresponding to the Rb_values list
colors = ['red','orange','green','blue']


for i in range(len(Rb_values)):
    params_value.update({'Rb':Rb_values[i]})

    sol1 = solve_ivp(funtion_model, [0, -300], [1.409, 0.85, 0.], args=(Rb_values[i],), max_step=.1)
    
    # plt.plot(np.array(sol1.y).swapaxes(0, 1))
    # plt.plot(sol1.y[0,:])
    # plt.title(f'{Rb_values[i]}')
    # plt.show()

    # Careful to adjust period timings depending on the other parameters
    # double check that the cycle is well captured

    period_start = 905
    period_end = 1314
    circle = np.concatenate([np.expand_dims(np.array(sol1.t[period_start:period_end]), 0),
                            np.array(sol1.y)[:, period_start:period_end]])
    
    # plt.figure()
    # plt.plot(circle[1], circle[2])
    # plt.show()

    HH = makeModel(0.01, params_value, ic_args)
    PC = pdst.ContClass(HH)

    PCargs = pdst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
    PCargs.freepars = ['L']                    # control parameter(s) (it should be among those specified in DSargs.pars)
    PCargs.StepSize = 8e-3
    PCargs.MaxNumPoints = 380
    PCargs.MaxStepSize = 4e-2
    PCargs.LocBifPoints = ["H"]
    PCargs.SaveEigen = True
    PCargs.verbosity = 2
    PC.newCurve(PCargs) 
    PC['EQ1'].forward()
    PC['EQ1'].backward()
    PC['EQ1'].display(['L','y'], stability=True, figure=1, color=colors[i])

    PCargs = pdst.args(name='LC1', type='LC-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
    
    # Create bifn curve for limit cycle
    PCargs.verbosity = 2
    PCargs.freepars = ['L']
    PCargs.initcycle = circle
    # PCargs.MinStepSize = 1e-6
    PCargs.MaxStepSize = 1e-2
    PCargs.StepSize = 1e-3
    PCargs.MaxNumPoints = 500
    PCargs.LocBifPoints = 'all'
    PCargs.StopAtPoints = 'B'
    PCargs.NumIntervals = 300
    PCargs.NumCollocation = 7
    # PCargs.FuncTol = 1e-11
    PCargs.TestTol = 1e-6
    # PCargs.VarTol = 1e-10
    PCargs.NumSPOut = 1000
    PCargs.SolutionMeasures = 'all'
    PCargs.SaveEigen = True
    PCargs.SaveFlow = True
    PCargs.SaveJacobian = True
    PC.newCurve(PCargs)
    print ('Computing curve...')
    PC['LC1'].forward()
    PC['LC1'].backward()
    PC['LC1'].display(('L', 'y'), stability=True, figure=1, points=True, color=colors[i])

# PC['LC1'].display(('L','y'), stability=True, figure=5)
# PC['LC1'].display(('L','x'), stability=True, figure=6)
# ax = plt.gca()
# for txt in ax.texts:
#     txt.set_visible(False)
# for i in range(len(Rb_values)):
#     plt.delaxes(ax.lines[10+7*i])
#     plt.delaxes(ax.lines[9+7*i])
#     plt.delaxes(ax.lines[8+7*i])
#     plt.delaxes(ax.lines[7+7*i])
#     ax.lines[6+7*i].set_label(f'Rb = {Rb_values[i]}')
# plt.legend()
plt.show()