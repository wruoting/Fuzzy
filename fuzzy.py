# we try to use c means clustering to set up
from __future__ import division, print_function
import numpy as np
import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
import colorlover as cl
import skfuzzy as fuzz

# TODO put a settings option to set colors for any data
def graph(x_axis_label, y_axis_label, x_y_import,
            colorscale='qual', colorscale_type='Paired', name = "Fuzzy_Test.html",
            options=False):
    data = []
    x_axis_label = ''
    y_axis_label = ''
    unique_colors = str(len(x_y_import)) if len(x_y_import) > 3 else '3'
    color_pie = cl.scales[unique_colors][colorscale][colorscale_type]
    # guide for color pie
    # https://plot.ly/ipython-notebooks/color-scales/
    if options:
        print(cl.scales[unique_colors])
    for index, set in enumerate(x_y_import):
        data.append(go.Scatter(
            x=set['x'],
            y=set['y'],
            mode='markers',
            name='Cluster '+str(index+1),
            marker = dict(
                size=5,
                color = color_pie[index]
            )
        ))
    fig= {
        'data': data,
        'layout': {
            'xaxis': {'title': x_axis_label},
            'yaxis': {'title': y_axis_label}
        }
    }

    py.offline.plot(fig,filename=name)


# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

all_data = np.vstack((xpts, ypts))

# What's array exponentiation applied to the membership function?
fpc_array = []
#len(all_data[0])
for index in range(1,9):
    index += 1
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        all_data, index, 2, error=0.005, maxiter=1000, init=None)
    fpc_array.append(fpc)
    #print(fpc)
print(fpc_array)
# FPC data
data = [{'x': list(range(2,10)), 'y': fpc_array}]
graph('Number of Clusters', 'FPC Scores', data,
    colorscale="div",
    colorscale_type="RdYlBu",
    name = "FPC.html")
max_fpc = np.argmax(fpc_array) + 2 # We add 2 here because 1 cluster is always 1, and we need to add 2 for the index

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        all_data, max_fpc, 2, error=0.005, maxiter=1000, init=None)

# Partition Data
data = []
for label in range(3):
    data.append({'x': xpts[labels == label], 'y': ypts[labels == label]})

# Centroid Data
for pt in cntr:
    data.append({'x': [pt[0]], 'y': [pt[1]]})
graph('X', 'Y', data)
