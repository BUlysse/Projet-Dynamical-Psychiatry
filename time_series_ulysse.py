import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
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

def ksi(t):
    ksi_value = np.random.normal(0,0.01)
    if ksi_value > 1:
        return 1
    if ksi_value < -1:
        return -1
    return ksi_value

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
            X[2] = -5
            pass
        Y[i,:] = X
    print(time.perf_counter() - start)
    return Y

def F(X,t): #X = [x,y,z,f]
    dx = 1/tx*(Smax/(1 + np.exp((Rs-X[1])/lambs)) - X[0])
    dy = 1/ty*(P/(1 + np.exp((Rb - X[1])/lambb)) + X[3]*L - X[0]*X[1] - X[2])
    dz = 1/tz*(S*(alpha*X[0] + beta*X[1])*(random.random() - 0.5) - X[2])
    df = 1/tf*(X[1] - lambf*X[3])
    return np.array([dx, dy, dz, df])

################CODE#######################

dt = 0.01
t = np.arange(0,8000,dt)

#A : Rb, L, S = 1.04, 0.2, 4
#B : Rb, L, S = 0.904, 0.2, 4
#C : Rb, L, S = 1.04, 1.01, 10
#D : Rb, L, S = 1, 0.6, 4.5

fig = make_subplots(rows=4, cols=1, vertical_spacing = 0.035, shared_xaxes=True, subplot_titles=("Healthy condition",""))


Rb, L, S = 1.04, 0.2, 4
Y = euler_modifie(F,X0,t)

fig.add_trace(
    go.Scatter(x=t, y=Y[:,0], mode='lines', line=dict(color="darkcyan",width=2), name="x_timecourse"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=t, y=Y[:,1], mode='lines', line=dict(color="magenta",width=2), name="y_timecourse"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=t, y=Y[:,2], mode='lines', line=dict(color="deepskyblue",width=2), name="z_timecourse"),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=t, y=Y[:,3], mode='lines', line=dict(color="coral",width=2), name="f_timecourse"),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        mode='markers',
        x=[5500],
        y=[4],
        marker=dict(
            color='black',
            size=200,
        ),
        marker_symbol='arrow-down'
    ),
    row=3, col=1
)

fig.update_xaxes(linecolor = 'black',linewidth = 2, row=1, col=1)
fig.update_xaxes(linecolor = 'black',linewidth = 2, row=2, col=1)
fig.update_xaxes(linecolor = 'black',linewidth = 2, row=3, col=1)
fig.update_xaxes(linecolor = 'black',linewidth = 2, row=4, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="x", linecolor = 'black',linewidth = 2, mirror = True, row=1, col=1)
fig.update_yaxes(title_text="y", linecolor = 'black',linewidth = 2, mirror = True, row=2, col=1)
fig.update_yaxes(title_text="z", linecolor = 'black',linewidth = 2, mirror = True, range=[-5, 5], row=3, col=1)
fig.update_yaxes(title_text="f", linecolor = 'black',linewidth = 2, mirror = True, range=[0, 1.5], row=4, col=1)

##PRINT FIGURE

layout2 = go.Layout(
    autosize=False,
    width=1360,
    height=800,
    showlegend=False,
    font=dict(family = "Arial",size = 16),

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