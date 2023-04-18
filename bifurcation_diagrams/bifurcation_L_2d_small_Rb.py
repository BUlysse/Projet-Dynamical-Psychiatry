import PyDSTool as pdst
from PyDSTool.Toolbox.phaseplane import *
import matplotlib.pyplot as plt


def makeModel(dt, params_value, ic_args):
    '''
    dt : simulations timestep
    params_value : parameters values
    ic_args : initial conditions for the variables
    '''
    #definition of model's equations
    xstr = '(Smax/(1+exp((Rs-y)/lambdas))-x)/taux'
    ystr = '(P/(1+exp((Rb-y)/lambdab))+L*f-y*x-z)/tauy'

    #simulations parameters
    DSargs = pdst.args()
    DSargs.varspecs = {'x': xstr, 'y': ystr}
    DSargs.xdomain = {'x': [0,10], 'y': [0,4] } #variables bounds
    DSargs.pars = params_value
    DSargs.algparams = {'init_step':dt, 'max_step': dt*1.5,'max_pts': 300000, 'refine': 1}
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = 'Dynamical Psychiatry'

    return pdst.Generator.Vode_ODEsystem(DSargs)


# ------------------------------------------------------------
import numpy as np

params_value = {'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':0.904, 'lambdab': 0.05, 'L':1.01, 'tauy':14, 'S':10, 'alpha':0.5, 'beta':0.5, 'tauz':1, 'lambdaf': 1, 'tauf': 720, 'a': 0}

params_value.update({'z':0, 'f':1, 'Rb':0.954, 'L':0})
ic_args = {'x': 0.15, 'y': 0.45}

HH = makeModel(0.01, params_value, ic_args)
HH.set(tdata=[0, 8000])
HH.set(pdomain={'L':[0, np.Infinity]})

PC = pdst.ContClass(HH)

PCargs = pdst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
PCargs.freepars = ['L']                    # control parameter(s) (it should be among those specified in DSargs.pars)
PCargs.StepSize = 8e-3
PCargs.MaxNumPoints = 400
PCargs.MaxStepSize = 4e-2
PCargs.LocBifPoints = ['H']
PCargs.SaveEigen = True
PCargs.verbosity = 2

PC.newCurve(PCargs)
PC['EQ1'].forward()

PC.display(('L','y'),stability=True, figure=1, domain=True)        # stable and unstable branches as solid and dashed curves, resp.
plt.xlim([0, 3])
plt.title('Bifurcation diagram for y variable')
PC.display(('L','x'),stability=True, figure=2, domain=True)
plt.xlim([0, 3])
plt.title('Bifurcation diagram for x variable')
PC.display(('y','x'),stability=True, figure=3)
plt.title('Y-X bifurcation diagram')
plt.show()