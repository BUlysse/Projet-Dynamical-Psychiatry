from dash import Dash, html, dcc, Input, Output
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import random
import copy

import PyDSTool as pdst
from PyDSTool.Toolbox.phaseplane import *

def makeModel(dt, par_args, ic_args):
    # extra_terms must not introduce new variables!
    xstr = '(Smax/(1+exp((Rs-y)/lambs))-x)/tx'
    ystr = '(P/(1+exp((Rb-y)/lambb))+L*f-y*x-z)/ty'
    zstr = '(S*(alpha*x+beta*y)*0-z)/tz'
    fstr = '(y-lambf*f)/tf'

    # Testing: ptest in RHS function to test Jacobian computation of ionic with embedded aux function
    # ptest has no numeric effect on the ionic function otherwise
    #auxdict = {'a': ([],'1/2')}

    DSargs = pdst.args()
    DSargs.varspecs = {'x': xstr, 'y': ystr, 'z': zstr, 'f': fstr}
    DSargs.pars = par_args
    DSargs.xdomain = {'x': [0,10], 'y': [0,4], 'z': [-5,5], 'f': [0,1.5]}
    DSargs.algparams = {'init_step':dt, 'max_step': dt*1.5,
                        'max_pts': 300000, 'refine': 1}

    DSargs.checklevel = 0
    DSargs.ics = ic_args
    DSargs.name = 'Dynamical Psychiatry'

    return pdst.Generator.Vode_ODEsystem(DSargs)

class System:

    def __init__(self):
        self.__tx = 14
        self.__Smax = 10
        self.__Rs = 1
        self.__lambs = 0.1

        self.__ty = 14
        self.__P = 10
        self.__Rb = 1.04
        self.__lambb = 0.05
        self.__L = 0.2

        self.__tz = 1
        self.__S = 4
        self.__alpha = 0.5
        self.__beta = 0.5

        self.__tf = 720
        self.__lambf = 1
        self.__trajectory = None
        self.__noise = True
        self.__perturbation = False

        self.__fixed_points = None

    def __F(self,X,dt):
        dx = 1/self.__tx*(self.__Smax/(1 + np.exp((self.__Rs-X[1])/self.__lambs)) - X[0])
        dy = 1/self.__ty*(self.__P/(1 + np.exp((self.__Rb - X[1])/self.__lambb)) + X[3]*self.__L - X[0]*X[1] - X[2])
        dz = 1/self.__tz*(self.__S*(self.__alpha*X[0] + self.__beta*X[1])*(random.random()-0.5)*0.01/dt*self.__noise - X[2])
        df = 1/self.__tf*(X[1] - self.__lambf*X[3])
        return np.array([dx, dy, dz, df])

    def set_Rb(self,Rb):
        self.__Rb = Rb

    def get_Rb(self):
        return self.__Rb

    def set_L(self,L):
        self.__L = L

    def get_L(self):
        return self.__L

    def set_Rs(self,Rs):
        self.__Rs = Rs

    def get_Rs(self):
        return self.__Rs

    def set_lambb(self,lambb):
        self.__lambb = lambb

    def get_lambb(self):
        return self.__lambb

    def set_noise(self,noise):
        self.__noise = noise

    def get_noise(self):
        return self.__noise

    def set_perturbation(self,perturbation):
        self.__perturbation = perturbation

    def get_perturbation(self):
        return self.__perturbation

    def make_y_cline(self,Y,z,f):
        return np.array([Y,(self.__P/(1 + np.exp((self.__Rb - Y)/self.__lambb)) + f*self.__L - z)/Y])

    def make_x_cline(self,Y):
        return np.array([Y,self.__Smax/(1 + np.exp((self.__Rs - Y)/self.__lambs))])

    def get_vector_field(self,z,f):
        def vect_field(X):
            dx = 1/self.__tx*(self.__Smax/(1 + np.exp((self.__Rs-X[1])/self.__lambs)) - X[0])
            dy = 1/self.__ty*(self.__P/(1 + np.exp((self.__Rb - X[1])/self.__lambb)) + f*self.__L - X[0]*X[1] - z)
            return np.array([dx, dy])
        return vect_field

    def get_jac_x_y(self,z,f):
        def jac_x_y(X):
            [x,y] = X
            return np.array([
                [-1.0/self.__tx, (-self.__Smax*(-1.0/self.__lambs)*np.exp((self.__Rs-y)/self.__lambs)*(1+np.exp((self.__Rs-y)/self.__lambs))**-2)/self.__tx],
                [-y/self.__ty, (-self.__P*(-1.0/self.__lambb)*np.exp((self.__Rb-y)/self.__lambb)*(1+np.exp((self.__Rb-y)/self.__lambb))**-2-x)/self.__ty]
                ])
        return jac_x_y

    def compute_trajectory(self,X0):

        X = copy.deepcopy(X0)
        LX = np.zeros((len(Lt),4))
        LX[0,:] = X

        for i in range(len(Lt)):
            dX = self.__F(X,dt)
            X = X + dX*dt
            LX[i,:] = X
            if self.__perturbation:
                if abs(Lt[i]-4000)<dt:
                    X[2]=-5

        self.__trajectory = LX
        return LX

    def get_trajectory(self):
        return self.__trajectory

    def compute_fixed_points_4D(self,X0={'x': 0, 'y': 0.1, 'z': 0, 'f': 0.0}):
        par_args = {'Smax':self.__Smax, 'Rs':self.__Rs, 'lambs':self.__lambs, 'tx':self.__tx, 'P': self.__P, 'Rb':self.__Rb, 'lambb': self.__lambb, 'L':self.__L, 'ty':self.__ty, 'S':self.__S, 'alpha':self.__alpha, 'beta':self.__beta, 'tz':self.__tz, 'lambf': self.__lambf, 'tf': self.__tf, 'a': 0}

        HH = makeModel(0.01, par_args, X0)
        HH.set(tdata=[0, 8000])

        jac, new_fnspecs = pdst.prepJacobian(HH.funcspec._initargs['varspecs'], ['x', 'y', 'z', 'f'])

        scope = copy.copy(HH.pars)
        scope.update(new_fnspecs)
        jac_fn = pdst.expr2fun(jac, ensure_args=['t'], **scope)

        fp_coords = find_fixedpoints(HH, n=25, jac=jac_fn, eps=1e-9,
                                    subdomain={'x':HH.xdomain['x'],'y':HH.xdomain['y'],
                                    'z': HH.xdomain['z'], 'f': HH.xdomain['f']})

        fps=[]
        for fp in fp_coords:
                fps.append(fixedpoint_nD(HH, pdst.Point(fp), coords=['x', 'y', 'z', 'f'],
                                jac=jac_fn, eps=1e-8))

        print('Coordonnées (x, y, z, f) des points fixes 4D trouvés :')
        for fp in fps:
            print("F.p. at (%.5f, %.5f, %.5f, %.5f) has stability %s" % (fp.point['x'], fp.point['y'], fp.point['z'], fp.point['f'], fp.stability))
        self.__fixed_points = fps
        return fps

    def compute_fixed_points_2D(self, z, f, X0={'x': 0, 'y': 0.1, 'z': 0, 'f': 0.0}):
        par_args = {'Smax':self.__Smax, 'Rs':self.__Rs, 'lambs':self.__lambs, 'tx':self.__tx, 'P': self.__P, 'Rb':self.__Rb, 'lambb': self.__lambb, 'L':self.__L, 'ty':self.__ty, 'S':self.__S, 'alpha':self.__alpha, 'beta':self.__beta, 'tz':self.__tz, 'lambf': self.__lambf, 'tf': self.__tf, 'a': 0}

        HH = makeModel(0.01, par_args, X0)
        HH.set(tdata=[0, 8000])

        jac, new_fnspecs = pdst.prepJacobian(HH.funcspec._initargs['varspecs'], ['x', 'y'])
        scope = copy.copy(HH.pars)
        scope.update({'z': z, 'f': f})
        scope.update(new_fnspecs)
        jac_fn = pdst.expr2fun(jac, ensure_args=['t'], **scope)

        fp_coords = find_fixedpoints(HH, n=25, jac=jac_fn, eps=1e-8,
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

    def set_fixed_points(self, fixed_points):
        self.__fixed_points = fixed_points

    def get_fixed_points(self):
        return self.__fixed_points


class Trajectory:

    def __init__(self):
        self.__X0 = None
        self.__trajectory = None

    def set_X0(self, X0):
        self.__X0 = X0

    def get_X0(self):
        return self.__X0

    def set_trajectory(self,trajectory):
        self.__trajectory = trajectory

    def get_trajectory(self):
        return self.__trajectory


##########################################################################
#########################################################################


def plotly_phase_plane(xlim, ylim, xcline, ycline, vect_field, trajectory, X0, fixed_points, print_trajectory):

    layout = go.Layout(
        autosize=False,
        width=670,
        height=650,
        title="X Y Phase plane",
        legend_title="",
        font=dict(family = "Arial",size = 18),
        legend=dict(orientation="h",
            yanchor="bottom",
            y=-0.26,
        ),

        xaxis= go.layout.XAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True,
                            range=ylim,
                            title='Y'),

        yaxis= go.layout.YAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True,
                            range=xlim,
                            title='X'),

        margin=go.layout.Margin(
            l=30,
            r=10,
            b=30,
            t=50,
            pad = 4
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    ####################################################################
    #----------------COMPUTE VECTOR FIELD-------------------------------
    ####################################################################

    # define a grid and compute direction at each point

    x_range = np.linspace(xlim[0], xlim[1], 35)
    y_range = np.linspace(ylim[0],ylim[1], 35)

    X1 , Y1  = np.meshgrid(x_range, y_range)                     # create a grid
    DX1, DY1 = vect_field([X1, Y1])
    M = (np.hypot(DX1, DY1))                           # Norm of the growth rate
    M[ M == 0] = 1.                                 # Avoid zero division errors
    DX1 /= M                                        # Normalize each arrows
    DY1 /= M

    fig = ff.create_quiver(Y1, X1, DY1, DX1, scale=0.10, arrow_scale=0.4, line=dict(
                        color='grey'),name='vector field')

    fig.update_layout(layout)

    ####################################################################
    #---------------------PLOT NULLCLINES-------------------------------
    ####################################################################


    fig.add_trace(go.Scatter(x=xcline[0,:], y=xcline[1,:], mode='lines', line=dict(color="blue",width=3), name="xcline" ))
    fig.add_trace(go.Scatter(x=ycline[0,:], y=ycline[1,:], mode='lines', line=dict(color="red",width=3), name="ycline"))

    ####################################################################
    #---------------------PLOT FIXED POINTS-----------------------------
    ####################################################################

    stable_x = []
    stable_y = []
    unstable_x = []
    unstable_y = []
    for fp in fixed_points:
        if fp.stability == 's':
            stable_x.append(fp.point['x'])
            stable_y.append(fp.point['y'])
        if fp.stability == 'u':
            unstable_x.append(fp.point['x'])
            unstable_y.append(fp.point['y'])

    fig.add_trace(go.Scatter(x=stable_y, y=stable_x, mode='markers', marker=dict(color="black",size=15), name='stable fp'))
    fig.add_trace(go.Scatter(x=unstable_y, y=unstable_x, mode='markers', marker=dict(line_color="black",color="white",size=12,line_width=3), name='unstable fp'))


    ####################################################################
    #---------------------PLOT TRAJECTORY-------------------------------
    ####################################################################

    if print_trajectory :
        fig.add_trace(go.Scatter(x=trajectory[:,1], y=trajectory[:,0] , mode='lines', line=dict(color="yellowgreen",width=2), name="trajectory"))
        fig.add_trace(go.Scatter(x=np.array(X0[1]), y=np.array(X0[0]) , mode='markers', marker=dict(line_color="orange",size=15,line_width=5,symbol="x-thin"), name="initial_point"))

    ####################################################################
    #---------------------PLOT TIME COURSES-----------------------------
    ####################################################################

    fig2 = make_subplots(rows=4, cols=1, vertical_spacing = 0.12, shared_xaxes=True)

    fig2.add_trace(
        go.Scatter(x=Lt, y=trajectory[:,0], mode='lines', line=dict(color="darkcyan",width=2), name="x_timecourse"),
        row=1, col=1
    )

    fig2.add_trace(
        go.Scatter(x=Lt, y=trajectory[:,1], mode='lines', line=dict(color="magenta",width=2), name="y_timecourse"),
        row=2, col=1
    )

    fig2.add_trace(
        go.Scatter(x=Lt, y=trajectory[:,2], mode='lines', line=dict(color="deepskyblue",width=2), name="z_timecourse"),
        row=3, col=1
    )

    fig2.add_trace(
        go.Scatter(x=Lt, y=trajectory[:,3], mode='lines', line=dict(color="coral",width=2), name="f_timecourse"),
        row=4, col=1
    )

    fig2.update_xaxes(linecolor = 'black',linewidth = 2, row=1, col=1)
    fig2.update_xaxes(linecolor = 'black',linewidth = 2, row=2, col=1)
    fig2.update_xaxes(linecolor = 'black',linewidth = 2, row=3, col=1)
    fig2.update_xaxes(title_text="t (jours)", linecolor = 'black',linewidth = 2, row=4, col=1)

    # Update yaxis properties
    fig2.update_yaxes(title_text="x", linecolor = 'black',linewidth = 2, mirror = True, row=1, col=1)
    fig2.update_yaxes(title_text="y", linecolor = 'black',linewidth = 2, mirror = True, row=2, col=1)
    fig2.update_yaxes(title_text="z", linecolor = 'black',linewidth = 2, mirror = True, range=[-5, 5], row=3, col=1)
    fig2.update_yaxes(title_text="f", linecolor = 'black',linewidth = 2, mirror = True, range=[0, 1.5], row=4, col=1)

    layout2 = go.Layout(
        autosize=False,
        width=560,
        height=650,
        title="Time course of x, y, z and f",
        showlegend=False,
        font=dict(family = "Arial",size = 16),

        margin=go.layout.Margin(
            l=10,
            r=10,
            b=30,
            t=50,
            pad = 20
        ),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig2.update_layout(layout2)

    return fig, fig2

#########################################################################################
###################################DASH APP##############################################
#########################################################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

heading = html.H4(
    "Dynamical Psychiatry model visualisation", className="bg-primary text-white p-2"
)

z_value = html.Div(
    [
        dbc.Label("Z value", html_for="z"),
        dcc.Slider(
                id="slider-z",
                min=-5,
                max=5,
                step=0.1,
                value=0,
                marks = {i:str(i) for i in range(-5,5,1)},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

f_value = html.Div(
    [
        dbc.Label("f value", html_for="f"),
        dcc.Slider(
                id="slider-f",
                min=0,
                max=1.8,
                step=0.01,
                value=0.5,
                marks = {i:str(i) for i in [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8]},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

l_value = html.Div(
    [
        dbc.Label("L value", html_for="l"),
        dcc.Slider(
                id="slider-L",
                min=0,
                max=2.5,
                step=0.01,
                value=0.2,
                marks = {i:str(i) for i in [0,0.5,1,1.5,2,2.5]},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

rb_value = html.Div(
    [
        dbc.Label("Rb value", html_for="rb"),
        dcc.Slider(
                id="slider-Rb",
                min=0.85,
                max=1.15,
                step=0.002,
                value=1.04,
                marks = {i:str(i) for i in np.round(np.linspace(0.85,1.15,5),2)},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

rs_value = html.Div(
    [
        dbc.Label("Rs value", html_for="Rs"),
        dcc.Slider(
                id="slider-Rs",
                min=0.2,
                max=2,
                step=0.05,
                value=1,
                marks = {i:str(i) for i in np.round(np.linspace(0.2,2,5),2)},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

lambb_value = html.Div(
    [
        dbc.Label("Lambda_b value", html_for="lambb"),
        dcc.Slider(
                id="slider-lambb",
                min=0,
                max=0.2,
                step=0.005,
                value=0.05,
                marks = {i:str(i) for i in np.linspace(0,0.2,6)},
                tooltip={"placement": "bottom", "always_visible": True}
                ),
    ],
    className="mt-2",
)

x0_value = html.Div(
    [
        dbc.Label("Initial coordinates", html_for="x0"),
        dbc.Col([
            dbc.Label("x:", html_for="x"),
            dcc.Input(
                    id="input-x", type='text', value='0', debounce=True,style={'width':'15.5%','margin-left':'3px','margin-right':'5px'}
                    ),
            dbc.Label("y:", html_for="y"),
            dcc.Input(
                    id="input-y",
                    value='0.1', type='text', debounce=True, style={'width':'15.5%','margin-left':'3px','margin-right':'5px'}
                    ),
            dbc.Label("z:", html_for="z"),
            dcc.Input(
                    id="input-z",
                    value='0', type='text', debounce=True, style={'width':'15.5%','margin-left':'3px','margin-right':'5px'}
                    ),
            dbc.Label("f:", html_for="f"),
            dcc.Input(
                    id="input-f",
                    value='0', type='text', debounce=True, style={'width':'15.5%','margin-left':'3px','margin-right':'0px'}
                    ),
        ])
    ],
    className="mt-2"
)

checklist = html.Div(
    [
        dcc.Checklist(
            id="options",
            options = ['Noise','Perturbation','Trajectory'],
            value = ['Noise'],
            inputStyle={"margin-left": "10px","margin-right": "4px"}
        )
    ],
    className="mt-2",
)

control_panel = dbc.Card(
    dbc.CardBody(
        [z_value,f_value,html.Hr(),l_value,rb_value,rs_value,lambb_value, html.Hr(),x0_value,html.Hr(),checklist],
        className="bg-light",
    )
)

graph = dbc.Card(
    dbc.Row([dbc.Col(dcc.Graph(id="phase")),dbc.Col(dcc.Graph(id="time_course"))])
)

app.layout = html.Div(
    [heading, dbc.Row([dbc.Col(control_panel, md=2), dbc.Col(graph, md=10)])]
)


@app.callback(
    Output("phase", "figure"),
    Output("time_course", "figure"),
    Input("slider-z", "value"),
    Input("slider-f", "value"),
    Input("slider-L", "value"),
    Input("slider-Rb", "value"),
    Input("slider-Rs", "value"),
    Input("slider-lambb", "value"),
    Input("options", "value"),
    Input("input-x", "value"),
    Input("input-y", "value"),
    Input("input-z", "value"),
    Input("input-f", "value"),
)
def get_figure(z,f,L,Rb,Rs,lambb,options,x0='0',y0='0.1',z0='0',f0='0'):
    noise = False
    perturbation = False
    print_trajectory = False
    if 'Noise' in options:
        noise = True
    if 'Perturbation' in options:
        perturbation = True
    if 'Trajectory' in options:
        print_trajectory = True

    X0 = np.array([float(x0),float(y0),float(z0),float(f0)])
    update_fixed_points = True
    if L != system.get_L() or Rb != system.get_Rb() or Rs != system.get_Rs() or lambb != system.get_lambb():
        system.set_L(L)
        system.set_Rb(Rb)
        system.set_Rs(Rs)
        system.set_lambb(lambb)

    if (X0 != trajectory_cache.get_X0()).any() or noise != system.get_noise() or perturbation != system.get_perturbation():
        system.set_noise(noise)
        system.set_perturbation(perturbation)
        trajectory_cache.set_X0(X0)
        update_fixed_points = False #if the change was not about parameters that have an impact on fixed points, to not compute them

    trajectory_cache.set_trajectory(system.compute_trajectory(trajectory_cache.get_X0()))

    fixed_points = system.get_fixed_points()
    if not fixed_points or update_fixed_points:
        fixed_points = system.compute_fixed_points_2D(z,f)
        system.set_fixed_points(fixed_points)

    ycline = system.make_y_cline(y_range_null,z,f)
    ycline[:,np.add(ycline[1,:]<xlim[0],ycline[1,:]>xlim[1])] = None

    xcline = system.make_x_cline(y_range_null)
    xcline[:,np.add(xcline[1,:]<xlim[0],xcline[1,:]>xlim[1])] = None

    vect_field = system.get_vector_field(z,f)

    return plotly_phase_plane(xlim, ylim, xcline, ycline, vect_field, trajectory_cache.get_trajectory(), X0, fixed_points, print_trajectory)


if __name__ == "__main__":
    global system, trajectory_cache, xlim, y_range, xcline, dt, Lt

    system = System()
    trajectory_cache = Trajectory()

    xlim = [0, 10]
    ylim = [0, 3.5]
    dt = 0.01
    Lt = np.arange(0,5500,dt)

    y_range_null = np.logspace(np.log10(0.01),np.log10(ylim[1]), 500)
    xcline = system.make_x_cline(y_range_null)
    xcline[:,np.add(xcline[1,:]<xlim[0],xcline[1,:]>xlim[1])] = None

    app.run_server(debug=True)