import numpy as np
import random
import copy
import time

import PyDSTool as pdst
from PyDSTool.Toolbox.phaseplane import *

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
    fstr = '(y-lambdaf*f)/tauf'

    #simulations parameters
    DSargs = pdst.args()
    DSargs.varspecs = {'x': xstr, 'y': ystr, 'z': zstr, 'f': fstr}
    DSargs.xdomain = {'x': [0,10], 'y': [0,4], 'z': [-5,5], 'f': [0,1.5]} #variables bounds
    DSargs.pars = params_value
    DSargs.algparams = {'init_step':dt, 'max_step': dt*1.5,'max_pts': 300000, 'refine': 1}
    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = 'Dynamical Psychiatry'

    return pdst.Generator.Vode_ODEsystem(DSargs)


def compute_fixed_points_4D(**kwargs):
    par_args = {'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':1.04, 'lambdab': 0.05, 'L':1.01, 'tauy':14, 'S':10, 'alpha':0.5, 'beta':0.5, 'tauz':1, 'lambdaf': 1, 'tauf': 720, 'a': 0}
    par_args.update(kwargs)

    ic_args = {'x': 0, 'y': 0.1, 'z': 0, 'f': 0.0}
    HH = makeModel(0.01, par_args, ic_args)
    HH.set(tdata=[0, 8000])

    jac, new_fnspecs = pdst.prepJacobian(HH.funcspec._initargs['varspecs'], ['x', 'y', 'z', 'f'])

    scope = copy.copy(HH.pars)
    scope.update(new_fnspecs)
    jac_fn = pdst.expr2fun(jac, ensure_args=['t'], **scope)

    fp_coords = find_fixedpoints(HH, n=4, jac=jac_fn, eps=1e-8,
                                subdomain={'x':HH.xdomain['x'],'y':HH.xdomain['y'],
                                'z': HH.xdomain['z'], 'f': HH.xdomain['f']})

    fps=[]
    for fp in fp_coords:
            fps.append(fixedpoint_nD(HH, pdst.Point(fp), coords=['x', 'y', 'z', 'f'],
                            jac=jac_fn, eps=1e-8))

    print('Coordonnées (x, y, z, f) des points fixes 4D trouvés :')
    for fp in fps:
        print("F.p. at (%.5f, %.5f, %.5f, %.5f) has stability %s" % (fp.point['x'], fp.point['y'], fp.point['z'], fp.point['f'], fp.stability))
    return fps


def compute_fixed_points_2D(z, f, **kwargs):
    par_args = {'Smax':10, 'Rs':1, 'lambdas':0.1, 'taux':14, 'P': 10, 'Rb':1.04, 'lambdab': 0.05, 'L':1.01, 'tauy':14, 'S':10, 'alpha':0.5, 'beta':0.5, 'tauz':1, 'lambdaf': 1, 'tauf': 720, 'a': 0}
    par_args.update(kwargs)

    ic_args = {'x': 0, 'y': 0.1, 'z': 0, 'f': 0.0}
    HH = makeModel(0.01, par_args, ic_args)
    HH.set(tdata=[0, 8000])

    jac, new_fnspecs = pdst.prepJacobian(HH.funcspec._initargs['varspecs'], ['x', 'y'])
    scope = copy.copy(HH.pars)
    scope.update({'z': z, 'f': f})
    scope.update(new_fnspecs)
    jac_fn = pdst.expr2fun(jac, ensure_args=['t'], **scope)

    fp_coords = find_fixedpoints(HH, n=5, jac=jac_fn, eps=1e-8,
                                subdomain={'x':HH.xdomain['x'],'y':HH.xdomain['y'],
                                'z': z, 'f': f})

    fps=[]
    for fp in fp_coords:
            fps.append(fixedpoint_2D(HH, pdst.Point(fp), coords=['x', 'y'],
                            jac=jac_fn, eps=1e-8))

    print('Coordonnées (y, x) des points fixes dans le plan X-Y :')
    for fp in fps:
        print("F.p. at (%.5f, %.5f) is a %s and has stability %s" % (fp.point['x'], fp.point['y'], fp.classification, fp.stability))
    return fps


start = time.perf_counter()
compute_fixed_points_4D(L = 0.2, S = 4, Rb = 0.92) #rajouter des valeurs spécifiques de paramètres, par défaut L = 1.01, S = 10, Rb = 1.04
print(time.perf_counter() - start)

start = time.perf_counter()
compute_fixed_points_2D(z = 0, f = 0.61, L = 0.2, S = 4, Rb = 0.92) #rajouter des valeurs spécifiques de paramètres, par défaut L = 1.01, S = 10, Rb = 1.04
print(time.perf_counter() - start)