import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import plotly.colors as colors
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

X0 = [0,0.1,0,0]

#################EQUATIONS################

def euler_modifie(F,X0,t):
    Y = np.zeros((len(t),4))
    Y[0,:] = X0
    X = copy.deepcopy(X0)
    start = time.perf_counter()
    for i in range(1,len(t)):
        X = X  + F(X ,t[i])*dt
        if abs(t[i]-5500)<=dt:
            #X[2] = -5
            pass
        Y[i,:] = X
    print(time.perf_counter() - start)
    return Y

def F(X,t): #X = [x,y,z,f]
    dx = 1/tx*(Smax/(1 + np.exp((Rs-X[1])/lambs)) - X[0])
    dy = 1/ty*(P/(1 + np.exp((Rb - X[1])/lambb)) + X[3]*L - X[0]*X[1] - X[2])
    dz = 1/tz*(S*(alpha*X[0] + beta*X[1])*(random.random() - 0.5)*0 - X[2])
    df = 1/tf*(X[1] - lambf*X[3])
    return np.array([dx, dy, dz, df])

################CODE#######################

dt = 0.03
t = np.arange(0,8000,dt)

#A : Rb, L, S = 1.04, 0.2, 4
#B : Rb, L, S = 0.904, 0.2, 4
#C : Rb, L, S = 1.04, 1.01, 10
#D : Rb, L, S = 1, 0.6, 4.5


Rb, L, S = 0.904, 0.2, 4

Rb_values = [0.904, 1, 1.04, 1.15]
L_values = np.linspace(0,2.5,50)

color = colors.DEFAULT_PLOTLY_COLORS

data = []
for j in range(len(Rb_values)):
    f_values = np.zeros_like(L_values)
    for i in range(len(L_values)):
        Rb = Rb_values[j]
        L = L_values[i]
        Y = euler_modifie(F,X0,t)
        f_values[i] = Y[-1,3]
    data.append(go.Scatter(x=L_values, y=np.multiply(f_values,L_values), mode='lines', line=dict(color=color[j],width=2), name=f"Rb = {Rb_values[j]}"))

fig = go.Figure(
    data=data
)

fig.update_xaxes(title_text="L", linecolor = 'black',linewidth = 2)

# Update yaxis properties
fig.update_yaxes(title_text="Final f.L", linecolor = 'black',linewidth = 2)

##PRINT FIGURE

layout2 = go.Layout(
    autosize=False,
    width=860,
    height=500,
    showlegend=True,
    font=dict(family = "Arial",size = 24),

    margin=go.layout.Margin(
        l=10,
        r=100,
        b=30,
        t=50,
        pad = 10
    ),
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.update_layout(layout2)

fig.show()