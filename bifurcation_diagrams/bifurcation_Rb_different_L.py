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

params_value = {'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':1.04, 'lambdab': 0.05, 'L':1.01, 'tauy':14, 'S':10, 'alpha':0.5, 'beta':0.5, 'tauz':1, 'lambdaf': 1, 'tauf': 720, 'a': 0}

params_value.update({'z':0, 'f':1})
ic_args = {'x': 1.08, 'y': 0.78}

L_values = [0.2, 0.6, 1, 1.5, 3]
colors = ['red','orange','green','blue', 'pink']

#x diagram
for i in range(len(L_values)):

    HH = makeModel(0.01, params_value, ic_args)
    HH.set(tdata=[0, 8000])
    HH.set(pars={'Rb':1.25, 'L':L_values[i]})

    PC = pdst.ContClass(HH)

    PCargs = pdst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
    PCargs.freepars = ['Rb']                    # control parameter(s) (it should be among those specified in DSargs.pars)
    PCargs.StepSize = 4e-3
    PCargs.MaxNumPoints = 1000
    PCargs.MaxStepSize = 8e-3
    PCargs.LocBifPoints = ['H']
    PCargs.SaveEigen = True
    PCargs.verbosity = 2

    PC.newCurve(PCargs)
    PC['EQ1'].forward()

    PC.display(('Rb','x'),stability=True, figure=1, color=colors[i])
    plt.title('Bifurcation diagram for x variable')

ax = plt.gca()
for txt in ax.texts:
    txt.set_visible(False)
for i in range(len(L_values)):
    plt.delaxes(ax.lines[10+7*i])
    plt.delaxes(ax.lines[9+7*i])
    plt.delaxes(ax.lines[8+7*i])
    plt.delaxes(ax.lines[7+7*i])
    ax.lines[6+7*i].set_label(f'L = {L_values[i]}')
plt.legend()
plt.show()