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