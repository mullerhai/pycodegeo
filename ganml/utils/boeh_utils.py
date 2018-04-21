import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin


from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import GnBu3, OrRd3
from bokeh.plotting import figure

from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral5
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg as df
from bokeh.transform import factor_cmap


from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Spectral5
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap

from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.sampledata.commits import data
from bokeh.transform import jitter

from numpy import linspace
from scipy.stats.kde import gaussian_kde

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly

import colorcet as cc

import pandas as pd

from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

import networkx as nx

from bokeh.io import show, output_file
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral4


from bokeh.plotting import figure, show, output_file
from bokeh.tile_providers import CARTODBPOSITRON

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import Toggle, BoxAnnotation, CustomJS

import numpy as np

from bokeh.models import BoxSelectTool, BoxZoomTool, LassoSelectTool
from bokeh.plotting import figure, output_file, show

import numpy as np
from bokeh.plotting import output_file, show, figure

import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.models import LogColorMapper, LogTicker, ColorBar

output_file('color_bar.html')

def normal2d(X, Y, sigx=1.0, sigy=1.0, mux=0.0, muy=0.0):
    z = (X-mux)**2 / sigx**2 + (Y-muy)**2 / sigy**2
    return np.exp(-z/2) / (2 * np.pi * sigx * sigy)

X, Y = np.mgrid[-3:3:100j, -2:2:100j]
Z = normal2d(X, Y, 0.1, 0.2, 1.0, 1.0) + 0.1*normal2d(X, Y, 1.0, 1.0)
image = Z * 1e6

color_mapper = LogColorMapper(palette="Viridis256", low=1, high=1e7)

plot = figure(x_range=(0,1), y_range=(0,1), toolbar_location=None)
plot.image(image=[image], color_mapper=color_mapper,
           dh=[1.0], dw=[1.0], x=[0], y=[0])

color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

plot.add_layout(color_bar, 'right')

show(plot)
#
# x = np.linspace(0, 4*np.pi, 100)
# y = np.sin(x)
#
# output_file("legend.html")
#
# p = figure()
#
# p.circle(x, y, legend="sin(x)")
# p.line(x, y, legend="sin(x)")
#
# p.line(x, 2*y, legend="2*sin(x)",
#        line_dash=[4, 4], line_color="orange", line_width=2)
#
# p.square(x, 3*y, legend="3*sin(x)", fill_color=None, line_color="green")
# p.line(x, 3*y, legend="3*sin(x)", line_color="green")
#
# show(p)
#
# output_file("styling_tool_overlays.html")
#
# x = np.random.random(size=200)
# y = np.random.random(size=200)
#
# # Basic plot setup
# plot = figure(plot_width=400, plot_height=400, title='Select and Zoom',
#               tools="box_select,box_zoom,lasso_select,reset")
#
# plot.circle(x, y, size=5)
#
# select_overlay = plot.select_one(BoxSelectTool).overlay
#
# select_overlay.fill_color = "firebrick"
# select_overlay.line_color = None
#
# zoom_overlay = plot.select_one(BoxZoomTool).overlay
#
# zoom_overlay.line_color = "olive"
# zoom_overlay.line_width = 8
# zoom_overlay.line_dash = "solid"
# zoom_overlay.fill_color = None
#
# plot.select_one(LassoSelectTool).overlay.line_dash = [10, 10]
#
# show(plot)
#
#
# # We set-up the same standard figure with two lines and now a box over top
# p = figure(plot_width=600, plot_height=200, tools='')
# visible_line = p.line([1, 2, 3], [1, 2, 1], line_color="blue")
# invisible_line = p.line([1, 2, 3], [2, 1, 2], line_color="pink")
#
# box = BoxAnnotation(left=1.5, right=2.5, fill_color='green', fill_alpha=0.1)
# p.add_layout(box)
#
# # We write coffeescript to link toggle with visible property of box and line
# code = '''\
# object.visible = toggle.active
# '''
#
# callback1 = CustomJS.from_coffeescript(code=code, args={})
# toggle1 = Toggle(label="Green Box", button_type="success", callback=callback1)
# callback1.args = {'toggle': toggle1, 'object': box}
#
# callback2 = CustomJS.from_coffeescript(code=code, args={})
# toggle2 = Toggle(label="Pink Line", button_type="success", callback=callback2)
# callback2.args = {'toggle': toggle2, 'object': invisible_line}
#
# output_file("styling_visible_annotation_with_interaction.html")
#
# show(layout([p], [toggle1, toggle2]))

#
#
# output_file("gmap.html")
#
# map_options = GMapOptions(lat=30.2861, lng=-97.7394, map_type="roadmap", zoom=11)
#
# # For GMaps to function, Google requires you obtain and enable an API key:
# #
# #     https://developers.google.com/maps/documentation/javascript/get-api-key
# #
# # Replace the value below with your personal API key:
# p = gmap("GOOGLE_API_KEY", map_options, title="Austin")
#
# source = ColumnDataSource(
#     data=dict(lat=[ 30.29,  30.20,  30.29],
#               lon=[-97.70, -97.74, -97.78])
# )
#
# p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)
#
# show(p)

#
# output_file("tile.html")
#
# # range bounds supplied in web mercator coordinates
# p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
#            x_axis_type="mercator", y_axis_type="mercator")
# p.add_tile(CARTODBPOSITRON)
#
# show(p)

#
# G=nx.karate_club_graph()
#
# plot = Plot(plot_width=400, plot_height=400,
#             x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
# plot.title.text = "Graph Interaction Demonstration"
#
# plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())
#
# graph_renderer = from_networkx(G, nx.circular_layout, scale=1, center=(0,0))
#
# graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
# graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
# graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
#
# graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
# graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
# graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
#
# graph_renderer.selection_policy = NodesAndLinkedEdges()
# graph_renderer.inspection_policy = EdgesAndLinkedNodes()
#
# plot.renderers.append(graph_renderer)
#
# output_file("interactive_graphs.html")
# show(plot)

#element
# output_file("periodic.html")
#
# periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
# groups = [str(x) for x in range(1, 19)]
#
# df = elements.copy()
# df["atomic mass"] = df["atomic mass"].astype(str)
# df["group"] = df["group"].astype(str)
# df["period"] = [periods[x-1] for x in df.period]
# df = df[df.group != "-"]
# df = df[df.symbol != "Lr"]
# df = df[df.symbol != "Lu"]
#
# cmap = {
#     "alkali metal"         : "#a6cee3",
#     "alkaline earth metal" : "#1f78b4",
#     "metal"                : "#d93b43",
#     "halogen"              : "#999d9a",
#     "metalloid"            : "#e08d49",
#     "noble gas"            : "#eaeaea",
#     "nonmetal"             : "#f1d4Af",
#     "transition metal"     : "#599d7A",
# }
#
# source = ColumnDataSource(df)
#
# p = figure(plot_width=900, plot_height=500, title="Periodic Table (omitting LA and AC Series)",
#            x_range=groups, y_range=list(reversed(periods)), toolbar_location=None, tools="")
#
# p.rect("group", "period", 0.95, 0.95, source=source, fill_alpha=0.6, legend="metal",
#        color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))
#
# text_props = {"source": source, "text_align": "left", "text_baseline": "middle"}
#
# x = dodge("group", -0.4, range=p.x_range)
#
# r = p.text(x=x, y="period", text="symbol", **text_props)
# r.glyph.text_font_style="bold"
#
# r = p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number", **text_props)
# r.glyph.text_font_size="8pt"
#
# r = p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name", **text_props)
# r.glyph.text_font_size="5pt"
#
# r = p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass", **text_props)
# r.glyph.text_font_size="5pt"
#
# p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")
#
# p.add_tools(HoverTool(tooltips = [
#     ("Name", "@name"),
#     ("Atomic number", "@{atomic number}"),
#     ("Atomic mass", "@{atomic mass}"),
#     ("Type", "@metal"),
#     ("CPK color", "$color[hex, swatch]:CPK"),
#     ("Electronic configuration", "@{electronic configuration}"),
# ]))
#
# p.outline_line_color = None
# p.grid.grid_line_color = None
# p.axis.axis_line_color = None
# p.axis.major_tick_line_color = None
# p.axis.major_label_standoff = 0
# p.legend.orientation = "horizontal"
# p.legend.location ="top_center"
#
# show(p)

#
# output_file("unemploymemt.html")
#
# data.Year = data.Year.astype(str)
# data = data.set_index('Year')
# data.drop('Annual', axis=1, inplace=True)
# data.columns.name = 'Month'
#
# # reshape to 1D array or rates with a month and year for each row.
# df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
#
# source = ColumnDataSource(df)
#
# # this is the colormap from the original NYTimes plot
# colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
# mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())
#
# p = figure(plot_width=800, plot_height=300, title="US Unemployment 1948—2016",
#            x_range=list(data.index), y_range=list(reversed(data.columns)),
#            toolbar_location=None, tools="", x_axis_location="above")
#
# p.rect(x="Year", y="Month", width=1, height=1, source=source,
#        line_color=None, fill_color=transform('rate', mapper))
#
# color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
#                      ticker=BasicTicker(desired_num_ticks=len(colors)),
#                      formatter=PrintfTickFormatter(format="%d%%"))
#
# p.add_layout(color_bar, 'right')
#
# p.axis.axis_line_color = None
# p.axis.major_tick_line_color = None
# p.axis.major_label_text_font_size = "5pt"
# p.axis.major_label_standoff = 0
# p.xaxis.major_label_orientation = 1.0
#
# show(p)
#
# output_file("joyplot.html")
#
# def joy(category, data, scale=20):
#     return list(zip([category]*len(data), scale*data))
#
# cats = list(reversed(probly.keys()))
#
# palette = [cc.rainbow[i*15] for i in range(17)]
#
# x = linspace(-20,110, 500)
#
# source = ColumnDataSource(data=dict(x=x))
#
# p = figure(y_range=cats, plot_width=700, x_range=(-5, 105), toolbar_location=None)
#
# for i, cat in enumerate(reversed(cats)):
#     pdf = gaussian_kde(probly[cat])
#     y = joy(cat, pdf(x))
#     source.add(y, cat)
#     p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)
#
# p.outline_line_color = None
# p.background_fill_color = "#efefef"
#
# p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
# p.xaxis.formatter = PrintfTickFormatter(format="%d%%")
#
# p.ygrid.grid_line_color = None
# p.xgrid.grid_line_color = "#dddddd"
# p.xgrid.ticker = p.xaxis[0].ticker
#
# p.axis.minor_tick_line_color = None
# p.axis.major_tick_line_color = None
# p.axis.axis_line_color = None
#
# p.y_range.range_padding = 0.12
#
# show(p)

#
# output_file("bars.html")
#
# DAYS = ['Sun', 'Sat', 'Fri', 'Thu', 'Wed', 'Tue', 'Mon']
#
# source = ColumnDataSource(data)
#
# p = figure(plot_width=800, plot_height=300, y_range=DAYS, x_axis_type='datetime',
#            title="Commits by Time of Day (US/Central) 2012—2016")
#
# p.circle(x='time', y=jitter('day', width=0.6, range=p.y_range),  source=source, alpha=0.3)
#
# p.xaxis[0].formatter.days = ['%Hh']
# p.x_range.range_padding = 0
# p.ygrid.grid_line_color = None
#
# show(p)
#
# output_file("bars.html")
#
# df.cyl = df.cyl.astype(str)
# df.yr = df.yr.astype(str)
#
# group = df.groupby(('cyl', 'mfr'))
#
# source = ColumnDataSource(group)
# index_cmap = factor_cmap('cyl_mfr', palette=Spectral5, factors=sorted(df.cyl.unique()), end=1)
#
# p = figure(plot_width=800, plot_height=300, title="Mean MPG by # Cylinders and Manufacturer",
#            x_range=group, toolbar_location=None, tools="")
#
# p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,
#        line_color="white", fill_color=index_cmap, )
#
# p.y_range.start = 0
# p.x_range.range_padding = 0.05
# p.xgrid.grid_line_color = None
# p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
# p.xaxis.major_label_orientation = 1.2
# p.outline_line_color = None
#
# p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))
#
# show(p)

#
# output_file("groupby.html")
#
# df.cyl = df.cyl.astype(str)
# group = df.groupby('cyl')
#
# source = ColumnDataSource(group)
#
# cyl_cmap = factor_cmap('cyl', palette=Spectral5, factors=sorted(df.cyl.unique()))
#
# p = figure(plot_height=350, x_range=group, title="MPG by # Cylinders",
#            toolbar_location=None, tools="")
#
# p.vbar(x='cyl', top='mpg_mean', width=1, source=source,
#        line_color=cyl_cmap, fill_color=cyl_cmap)
#
# p.y_range.start = 0
# p.xgrid.grid_line_color = None
# p.xaxis.axis_label = "some stuff"
# p.xaxis.major_label_orientation = 1.2
# p.outline_line_color = None
#
#
# output_file("stacked_split.html")
#
# fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
# years = ["2015", "2016", "2017"]
#
# exports = {'fruits' : fruits,
#            '2015'   : [2, 1, 4, 3, 2, 4],
#            '2016'   : [5, 3, 4, 2, 4, 6],
#            '2017'   : [3, 2, 4, 4, 5, 3]}
# imports = {'fruits' : fruits,
#            '2015'   : [-1, 0, -1, -3, -2, -1],
#            '2016'   : [-2, -1, -3, -1, -2, -2],
#            '2017'   : [-1, -2, -1, 0, -2, -2]}
#
# p = figure(y_range=fruits, plot_height=250, x_range=(-16, 16), title="Fruit import/export, by year",
#            toolbar_location=None)
#
# p.hbar_stack(years, y='fruits', height=0.9, color=GnBu3, source=ColumnDataSource(exports),
#              legend=["%s exports" % x for x in years])
#
# p.hbar_stack(years, y='fruits', height=0.9, color=OrRd3, source=ColumnDataSource(imports),
#              legend=["%s imports" % x for x in years])
#
# p.y_range.range_padding = 0.1
# p.ygrid.grid_line_color = None
# p.legend.location = "top_left"
# p.axis.minor_tick_line_color = None
# p.outline_line_color = None
#
# show(p)



# output_file("colormapped_bars.html")
#
# fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
# counts = [5, 3, 4, 2, 4, 6]
#
# source = ColumnDataSource(data=dict(fruits=fruits, counts=counts, color=Spectral6))
#
# p = figure(x_range=fruits, y_range=(0,9), plot_height=250, title="Fruit Counts",
#            toolbar_location=None, tools="")
#
# p.vbar(x='fruits', top='counts', width=0.9, color='color', legend="fruits", source=source)
#
# p.xgrid.grid_line_color = None
# p.legend.orientation = "horizontal"
# p.legend.location = "top_center"
#
# show(p)

#
# output_file("linked_selection_subsets.html")
#
# x = list(range(-20, 21))
# y0 = [abs(xx) for xx in x]
# y1 = [xx**2 for xx in x]
#
# # create a column data source for the plots to share
# source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))
#
# # create a view of the source for one plot to use
# view = CDSView(source=source, filters=[BooleanFilter([True if y > 250 or y < 100 else False for y in y1])])
#
# TOOLS = "box_select,lasso_select,hover,help"
#
# # create a new plot and add a renderer
# left = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
# left.circle('x', 'y0', size=10, hover_color="firebrick", source=source)
#
# # create another new plot, add a renderer that uses the view of the data source
# right = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
# right.circle('x', 'y1', size=10, hover_color="firebrick", source=source, view=view)
#
# p = gridplot([[left, right]])
#
# show(p)

#
# output_file("boolean_filter.html")
#
# source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]))
# booleans = [True if y_val > 2 else False for y_val in source.data['y']]
# view = CDSView(source=source, filters=[BooleanFilter(booleans)])
#
# tools = ["box_select", "hover", "reset"]
# p = figure(plot_height=300, plot_width=300, tools=tools)
# p.circle(x="x", y="y", size=10, hover_color="red", source=source)
#
# p_filtered = figure(plot_height=300, plot_width=300, tools=tools,
#                     x_range=p.x_range, y_range=p.y_range)
# p_filtered.circle(x="x", y="y", size=10, hover_color="red", source=source, view=view)
#
# show(gridplot([[p, p_filtered]]))
#
# import numpy as np
#
# from bokeh.plotting import figure, output_file, show
#
# # create an array of RGBA data
# N = 20
# img = np.empty((N, N), dtype=np.uint32)
# view = img.view(dtype=np.uint8).reshape((N, N, 4))
# for i in range(N):
#     for j in range(N):
#         view[i, j, 0] = int(255 * i / N)
#         view[i, j, 1] = 158
#         view[i, j, 2] = int(255 * j / N)
#         view[i, j, 3] = 255
#
# output_file("image_rgba.html")
#
# p = figure(plot_width=400, plot_height=400, x_range=(0, 10), y_range=(0, 10))
#
# p.image_rgba(image=[img], x=[0], y=[0], dw=[10], dh=[10])
#
# show(p)
#
# n = 50000
# x = np.random.standard_normal(n)
# y = np.random.standard_normal(n)
#
# bins = hexbin(x, y, 0.1)
#
# p = figure(tools="wheel_zoom,reset", match_aspect=True, background_fill_color='#440154')
# p.grid.visible = False
#
# p.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,
#            fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))
#
# output_file("hex_tile.html")
#
#show(p)