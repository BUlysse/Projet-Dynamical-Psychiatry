import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

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
    #print(time.perf_counter() - start)
    return Y

def F(X,t): #X = [x,y,z,f]
    dx = 1/tx*(Smax/(1 + np.exp((Rs-X[1])/lambs)) - X[0])
    dy = 1/ty*(P/(1 + np.exp((Rb - X[1])/lambb)) + X[3]*L - X[0]*X[1] - X[2])
    dz = 1/tz*(S*(alpha*X[0] + beta*X[1])*(random.random() - 0.5) - X[2])
    df = 1/tf*(X[1] - lambf*X[3])
    return np.array([dx, dy, dz, df])

################CODE#######################

dt = 0.01
t = np.arange(0,6000,dt)

#A : Rb, L, S = 1.04, 0.2, 4
#B : Rb, L, S = 0.904, 0.2, 4
#C : Rb, L, S = 1.04, 1.01, 10
#D : Rb, L, S = 1, 0.6, 4.5

Rb, L, S = 1.04, 0.2, 4

L_values = np.linspace(0.75,2.5,10)
count_values = np.zeros_like(L_values)
means_x = np.zeros_like(L_values)
means_y = np.zeros_like(L_values)

def count_oscillations(Y):
    x = Y[:,0]
    count = 0
    i=0
    under_8 = True
    while i < len(x):
        if x[i]<8 and not under_8:
            under_8 = True
        elif x[i]>8 and under_8:
            count+=1
            under_8 = False
        i+=1
    return count

for i in tqdm(range(len(L_values))):
    L = L_values[i]
    Y = euler_modifie(F,X0,t)
    count_values[i] = count_oscillations(Y)
    means_x[i] = np.mean(Y[int(5000/dt):int(6000/dt),0])
    means_y[i] = np.mean(Y[int(5000/dt):int(6000/dt),1])

fig = go.Figure(
    data=[go.Scatter(x=L_values, y=means_x, mode='lines', line=dict(color="blue",width=2), name="mean of x values"),
          go.Scatter(x=L_values, y=means_y, mode='lines', line=dict(color="red",width=2), name="mean of y values")]
)

fig.update_xaxes(title_text="L", linecolor = 'black',linewidth = 2)

# Update yaxis properties
fig.update_yaxes(title_text="Number of periods", linecolor = 'black', linewidth = 2, mirror = True)

##PRINT FIGURE

layout2 = go.Layout(
    autosize=False,
    width=860,
    height=500,
    showlegend=False,
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