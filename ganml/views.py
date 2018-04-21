from django.shortcuts import render

from ganml.utils.hive_utils import hive_client
from rest_pandas import PandasView
from .models import  TimeSeries
from .serializers import TimeSeriesSerializer
# Create your views here.
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, CDSView, IndexFilter
from bokeh.layouts import gridplot
from bokeh.embed import components
from bokeh.resources import CDN

from rest_pandas import PandasSimpleView
import  pandas as pd

class TimeSeriesView(PandasSimpleView):

  def get_data(self,request,*args,**kwargs):
    return pd.read_csv('')

class TimeSeriesView(PandasView):
    # Assign a default model queryset to the view
    queryset = TimeSeries.objects.all()

    def filter_queryset(self, qs):
      return qs

    serializer_class = TimeSeriesSerializer

    def transform_dataframe(self, dataframe):
      dataframe.some_pivot_function(in_place=True)
      return dataframe

def hello(request):
  context={}
  context['hello']='guodegang'
  context['condition']=False
  context['host']='127.0.0.1'
  context['port']=4200
  context['user']='zhuzheng'
  context['pwd']='abc123.'

  source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5]))
  view = CDSView(source=source, filters=[IndexFilter([0, 2, 4])])

  tools = ["box_select", "hover", "reset"]
  p = figure(plot_height=300, plot_width=300, tools=tools)
  p.circle(x="x", y="y", size=10, hover_color="red", source=source)

  p_filtered = figure(plot_height=300, plot_width=300, tools=tools)
  p_filtered.circle(x="x", y="y", size=10, hover_color="red", source=source, view=view)

  # TOOLS = "hover,crosshair,pan,wheel_zoom,box_zoom,reset,save,box_select"
  # picture = figure(width=1200, height=400, tools=TOOLS)
  # picture.line(data['order'], data['value'], color='blue', alpha=0.5)
  #
  script, div = components(p, CDN)

  print(script)
  print(div)
  context['script'] = script
  context['div'] = div
  return render(request,'html/index.html',context)

def connect(request):

  print("connect hive")
  print(request.POST)
  dic=request.POST
  host=dic['host']
  port=int(dic['port'])
  user=dic['user']
  pwd=dic['pwd']
  print(host +str(port)+user+pwd)
  # cli =hive_client(host,user,pwd,port)
  # conn= cli.connhive()
  # df= cli.query_selectFileds_Dataframe(conn,'tab_client_label',["realname",'gid','card'],{'client_nmbr':'AA75','batch':'p1'},5)
  #
  path='/Users/geo/Documents/pycode/xgb/geo/AA02p7_new.txt'
  df=pd.read_csv(path,sep='\t',header=None)
  name=df.head
  context = {}
  context['name'] = name
  return  render(request,'html/connect.html',context)