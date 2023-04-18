# Dynamical Psychiatry Simulations

## Set Up

Create a new virtual environment with Python <= 3.9.13 (python 3.10 is not supported for PyDSTool)

```
conda create -n "DyPsy" python=3.9.13 ipython
```

Install packages

```
conda install -c conda-forge pydstool matplotlib dash itsdangerous==2.0.1 werkzeug==2.0.3 dash-bootstrap-components
```

## Usage

To run a program named `my_program.py` use :

```
python -m my_program
```

## Visualiaztion dashboard

If you want to use the visualization dashboard, run `dash_figure.py` :

```
python -m dash_figure
```

The following message will then appear :

```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app "dash_figure" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
```

Just click on the http link and the dashboard will appear in your navigator.

Every time you change a parameter value using the sliders, you will have to wait a bit till the figures update, just be patient !
(The associated navigator tab will be named "Updating ..." if the update did not occur yet)

## Programs function

Those program mainly print figures I used in my report.

* `bifurcation_diagrams` : print bifurcation diagram of the (x-y) subsystem for specific conditions
* `dash_figure.py` : run visualization tool
* `f_L_relation.py` : print the figure relating the final value of f.L and L
* `fixed_points.py` : compute the 2D and 4D fixed point of the system according to specified parameter values
* `limit_cycle.py` : [NOT FUNCTIONNAL] print the limit cycles of the system
* `phase_plane_2d.py` : print the (x-y) phase plane with the 2d fixed point (corresponding to the phase plane fixed points)
* `phase_plane.py` : print the (x-y) phase plane with the 4d fixed points (fixed points will not correspond to the 2d phase plane fixed points)
* `reproduce_clinical_cases.py` : print a figure with a simulation corresponding to each of the 4 clinical cases of the original article
* `time_series_damien.py` : code damien originally sent me to compute and print the times series
* `times_series_ulysse.py` : code I created to compute and print the time series
* `z_oscillations.py` : compute and print the number of oscillations between high and low symptomes as a function of L
