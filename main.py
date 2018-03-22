try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print ("WARNING: Cache for this example is available on Python 3 only.")
    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

from os.path import dirname, join

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, CDSView, BooleanFilter,PanTool,WheelZoomTool,SaveTool,ResetTool,Label
from bokeh.models.widgets import PreText, Select, Tabs, Panel
from bokeh.plotting import figure
import numpy as np
from scipy.stats import chi2_contingency

#DATA_DIR = "/home/cloud-user/taito_wrk/DONOTREMOVE/visualization_paper_for_ICAME/"
DEFAULT_TICKERS= ['FAREWELL','HUCK','BROWN-A','BROWN-B','BROWN-C','BROWN-D',
                'BROWN-E','BROWN-TOTAL','FROWN-TOTAL']

DEFAULT_TEXTS= ['FAREWELL.txt','HUCK.txt','BROWN-A.txt','BROWN-B.txt','BROWN-C.txt','BROWN-D.txt',
                'BROWN-E.txt','BROWN-TOTAL.txt','FROWN-TOTAL.txt']


def nix(val, lst):
    return [x for x in lst if x != val]

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient
    
@lru_cache()
def load_ticker(ticker):
    fname = '%s.txt' % ticker
    data = pd.read_csv(fname, header=None,
                       names=['freq','word','na'],sep="\t",index_col=0)
    data=data.drop('na', axis=1)
    data["rel"]=data["freq"]*10000/data["freq"].sum()
    data["rank"]=data.index
    #data = data.set_index('date')
    #return pd.DataFrame({ticker: data.c, ticker+'_returns': data.c.diff()})
    return data

@lru_cache()
def get_data(t1, t2):
    df1 = load_ticker(t1)
    df2 = load_ticker(t2)
    data = pd.merge(df1, df2, on='word',how='outer').fillna(0)
    data = data.dropna()
    data["sum_x"]=data["freq_x"].sum()
    data["sum_y"]=data["freq_y"].sum()
    def chisq(data):
      a= np.array([[data["freq_x"],data["freq_y"]],[data["sum_x"],data["sum_y"]]])
      return(pd.Series(chi2_contingency(a, lambda_="log-likelihood")[0:2],))
    data=pd.concat([data,pd.DataFrame(data.apply(chisq,axis=1))],axis=1)
    data.columns = data.columns.astype(str)
    data=data.rename(index=str, columns={"0": "LL", "1": "pval"})
    data["color"] = np.where(data["pval"] < 0.05, "red", "blue")
    return data

# set up widgets

stats = PreText(text='', width=500)
ticker1 = Select(value='HUCK', options=nix('FAREWELL', DEFAULT_TICKERS))
ticker2 = Select(value='FAREWELL', options=nix('HUCK', DEFAULT_TICKERS))
#vis_menu = Select(value='linear',options=nix('log',DEFAULT_VIS))

#set up joint df

source = ColumnDataSource(data=dict(word=[],rank_x=[],rank_y=[],freq_x=[],freq_y=[],
				    sum_x=[],sum_y=[],rel_x=[],rel_y=[],LL=[],pval=[],color=[]))

#source_static  = ColumnDataSource(data=dict(rank=[],freq=[], word=[], rel=[]))

#df=pd.merge(source, source_static, on='word')
# set up plots

#source = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
#source_static = ColumnDataSource(data=dict(date=[], t1=[], t2=[], t1_returns=[], t2_returns=[]))
custom_hover= HoverTool()

custom_hover.tooltips = """
    <style>
        .bk-tooltip>div:not(:first-child) {display:none;}
    </style>

    <b>word: </b> @word <br>
    <b>rank: </b> @rank_x <br>
    <b>freq: </b> @freq_x <br>
    <b>per_10k: </b> @rel_x <br>
    <b>log_l: </b> @LL <br>
    <b>p_val: </b> @pval <br>
"""


TOOLS = "box_select,help,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
#TOOLS = "box_select,help,pan,wheel_zoom,box_zoom,reset,previewsave"
#TOOLS = [BoxSelectTool(),PanTool(),custom_hover,WheelZoomTool(),SaveTool(),ResetTool()]
#TOOLS_r = [BoxSelectTool(),PanTool(),custom_hover_right,WheelZoomTool(),SaveTool(),ResetTool()]

panels_l = []
panels_r = []

for l_axis_type in ["linear", "log"]:
  # create a new plot and add a renderer
  left = figure(tools=TOOLS, plot_width=300, plot_height=300,x_axis_type=l_axis_type, y_axis_type=l_axis_type)
  left.circle('rank_x', 'rel_x', source=source,alpha=0.6, size=6, color='color',selection_color="orange")
  #label = Label(x=1.1, y=18, text="FUGG", text_font_size='30pt', text_color='#eeeeee')
  #left.add_layout(label)
  hover = left.select(dict(type=HoverTool))

  hover.tooltips={"word": "@word","rank":"@rank_x","freq":"@freq_x","per_10k":"@rel_x","LL":"@LL","pval":"@pval"}
  #left.add_tools(custom_hover)
  panel = Panel(child=left, title=l_axis_type)
  panels_l.append(panel)

for r_axis_type in ["linear", "log"]:  
  # create another new plot and add a renderer
  right = figure(tools=TOOLS, plot_width=300, plot_height=300,x_axis_type=r_axis_type, y_axis_type=r_axis_type)
  right.circle('rank_y', 'rel_y', source=source,alpha=0.6, size=6, color='color',selection_color="orange")
  hover = right.select(dict(type=HoverTool))
  hover.tooltips={"word": "@word","rank":"@rank_y","freq":"@freq_y","per_10k":"@rel_y","LL":"@LL","pval":"@pval"}
  #right.add_tools(custom_hover)
  panel = Panel(child=right, title=r_axis_type)
  panels_r.append(panel)
  
tabs_l = Tabs(tabs=panels_l)
tabs_r = Tabs(tabs=panels_r)

p = gridplot([[tabs_l, tabs_r]])
#show(p)
#ts1 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
#ts1.line('date', 't1', source=source_static)
#ts1.circle('date', 't1', size=1, source=source, color=None, selection_color="orange")

#ts2 = figure(plot_width=900, plot_height=200, tools=tools, x_axis_type='datetime', active_drag="xbox_select")
#ts2.x_range = ts1.x_range
#ts2.line('date', 't2', source=source_static)
#ts2.circle('date', 't2', size=1, source=source, color=None, selection_color="orange")

# set up callbacks

def ticker1_change(attrname, old, new):
    ticker2.options = nix(new, DEFAULT_TICKERS)
    update()

def ticker2_change(attrname, old, new):
    ticker1.options = nix(new, DEFAULT_TICKERS)
    update()

def update(selected=None):
    t1, t2 = ticker1.value, ticker2.value

    data = get_data(t1, t2)
    source.data = source.from_df(data[['word','rank_x','rank_y','freq_x','freq_y','sum_x','sum_y','rel_x','rel_y','LL','pval','color']])
    #source_static.data = source.data
    update_stats(data, t1, t2)
    left.title.text = '%s, Gini coef. = %s' % (t1, round(gini(source.data["freq_x"]),3))
    right.title.text = '%s, Gini coef. = %s' % (t2, round(gini(source.data["freq_y"]),3))
    #ts1.title.text, ts2.title.text = t1, t2

def update_stats(data, t1, t2):
    stats.text = str(round(data[['rank_x','rank_y','freq_x','freq_y']].describe(),2))

ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)

def selection_change(attrname, old, new):
    t1, t2 = ticker1.value, ticker2.value
    data = get_data(t1, t2)
    selected = source.selected['1d']['indices']
    if selected:
        data = data.iloc[selected, :]
    update_stats(data, t1, t2)

source.on_change('selected', selection_change)

# set up layout
widgets = column(ticker1, ticker2, stats)
main_row = row(p, widgets)
#series = column(ts1, ts2)
layout = column(main_row)

# initialize
update()

curdoc().add_root(layout)
curdoc().title = "Stocks"
