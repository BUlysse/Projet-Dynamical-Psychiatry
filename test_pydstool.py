from PyDSTool import *
import time

# Declare names and initial values for (symbolic) parameters
mu = Par(0.8, 'mu')
k = Par(7, 'k')
D = Par(0.5, 'D')
theta = Par(1, 'theta')
h = Par(0.5, 'h')

# Compute nontrivial boundary equilibrium initial condition from parameters (see reference for derivation)
v1_0 = mu*(2*D + mu) / (D*(1-h*mu-theta*h*mu) + mu*(1-h*mu))
v2_0 = 0.0
p1_0 = (1+h*v1_0)*(1-v1_0/k)
p2_0 = (D/(D+mu))*(1+theta*h*v1_0)*(1-v1_0/k)

# Declare symbolic variables
v1 = Var('v1')
v2 = Var('v2')
p1 = Var('p1')
p2 = Var('p2')

# Create Symbolic Quantity objects for definitions
v1rhs = v1*(1-v1/k) - v1*p1/(1+h*v1)
v2rhs = v2*(1-v2/k) - v2*p2/(1+h*v2)
p1rhs = -1*mu*p1 + v1*p1/(1+h*v1) + D*(((1+theta*h*v2)/(1+h*v2))*p2 - ((1+theta*h*v1)/(1+h*v1))*p1)
p2rhs = -1*mu*p2 + v2*p2/(1+h*v2) + D*(((1+theta*h*v1)/(1+h*v1))*p1 - ((1+theta*h*v2)/(1+h*v2))*p2)

# Build Generator
DSargs = args(name='PredatorPrey')
DSargs.pars = [mu, k, D, theta, h]
DSargs.varspecs = args(v1=v1rhs,
                       v2=v2rhs,
                       p1=p1rhs,
                       p2=p2rhs)
# Use eval method to get a float value from the symbolic definitions given in
# terms of parameter values
DSargs.ics = args(v1=v1_0.eval(), v2=v2_0, p1=p1_0.eval(), p2=p2_0.eval())
ode = Generator.Vode_ODEsystem(DSargs)

#Set up continuation class
PC = ContClass(ode)

PCargs = args(name='EQ1', type='EP-C')
PCargs.freepars = ['k']
PCargs.StepSize = 1e-2
PCargs.MaxNumPoints = 50
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True
PCargs.verbosity = 2
PC.newCurve(PCargs)

print('Computing curve...')
start = time.perf_counter()
PC['EQ1'].forward()
print('done in %.3f seconds!' % (time.perf_counter()-start))

PCargs.name = 'HO1'
PCargs.type = 'H-C2'
PCargs.initpoint = 'EQ1:H1'
PCargs.freepars = ['k','D']
PCargs.MaxNumPoints = 50
PCargs.MaxStepSize = 0.1
PCargs.LocBifPoints = ['ZH']
PCargs.SaveEigen = True
PC.newCurve(PCargs)

print('Computing Hopf curve...')
start = time.perf_counter()
PC['HO1'].forward()
print('done in %.3f seconds!' % (time.perf_counter()-start))

PCargs = args(name = 'FO1', type = 'LP-C')
PCargs.initpoint = 'HO1:ZH1'
PCargs.freepars = ['k','D']
PCargs.MaxNumPoints = 25
PCargs.MaxStepSize = 0.1
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True
PC.newCurve(PCargs)

# Plot bifurcation diagram
PC.display(('k','D'), stability=True, figure=1)
plt.title('Bifurcation diagram of equilibria in (k,D) parameters')
plt.show()