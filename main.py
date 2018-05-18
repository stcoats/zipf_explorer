

from os.path import dirname, join

import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.models.widgets import RadioButtonGroup, Slider, DataTable, TableColumn
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, CDSView, BooleanFilter,PanTool,WheelZoomTool,SaveTool,ResetTool,Label
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import PreText, Select, Tabs, Panel, Paragraph
from bokeh.plotting import figure
import numpy as np
from scipy.stats import chi2_contingency
#from nltk.corpus import stopwords

stopw_file = join(dirname(__file__),'data', 'google_200words.txt')
words=open(stopw_file).read()
ss=words.split()

DEFAULT_TICKERS= ['A_Farewell_to_Arms','Huckleberry_Finn',
 'austen-emma',
 'austen-persuasion',
 'austen-sense',
 'bible-kjv',
 'blake-poems',
 'bryant-stories',
 'burgess-busterbrown',
 'carroll-alice',
 'chesterton-ball',
 'chesterton-brown',
 'chesterton-thursday',
 'edgeworth-parents',
 'melville-moby_dick',
 'milton-paradise',
 'shakespeare-caesar',
 'shakespeare-hamlet',
 'shakespeare-macbeth',
 'whitman-leaves',
 'inaugural',
 'Brown_adventure',
 'Brown_belles_lettres',
 'Brown_editorial',
 'Brown_fiction',
 'Brown_government',
 'Brown_hobbies',
 'Brown_humor',
 'Brown_learned',
 'Brown_lore',
 'Brown_mystery',
 'Brown_news',
 'Brown_religion',
 'Brown_reviews',
 'Brown_romance',
 'Brown_science_fiction',
 'Brown_TOTAL',
 'Frown_TOTAL']


#DEFAULT_FILES= ['FAREWELL.txt','HUCK.txt','BROWN-A.txt','BROWN-B.txt','BROWN-C.txt','BROWN-D.txt',
#                'BROWN-E.txt','BROWN-TOTAL.txt','FROWN-TOTAL.txt']


def nix(val, lst):
    return [x for x in lst if x != val]

def chisq(data):
      a= np.array([[data["freq_x"],data["freq_y"]],[data["sum_x"],data["sum_y"]]])
      return(pd.Series(chi2_contingency(a, lambda_="log-likelihood")[0:2],))
      
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

def ent(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    entropy=scipy.stats.entropy(p_data)  # input probabilities to get the entropy 
    return entropy
  
def simp(data):
    p_data= data.value_counts()/len(data) # calculates the probabilities
    simp = p_data**2
    simp = 1-simp.sum()
    return simp

#@lru_cache()
def load_ticker(ticker):
    fname = join(dirname(__file__),'data', '%s.csv' % ticker)
    data = pd.read_csv(fname,index_col=0)
    #data=data.drop('na', axis=1)
    data["rel"]=data["freq"]*10000/data["freq"].sum()
    data["rank"]=data.index
    #data = data.set_index('date')
    #return pd.DataFrame({ticker: data.c, ticker+'_returns': data.c.diff()})
    return data

#@lru_cache()
def get_data(t1, t2):
    df1 = load_ticker(t1)
    df2 = load_ticker(t2)
    data = pd.merge(df1, df2, on='word',how='outer').fillna(0)
    data = data.dropna()
    data["rel_diff"]=round(data["rel_x"]-data["rel_y"],3)
    data["sum_x"]=data["freq_x"].sum()
    data["sum_y"]=data["freq_y"].sum()
    data=pd.concat([data,pd.DataFrame(data.apply(chisq,axis=1))],axis=1)
    data.columns = data.columns.astype(str)
    data=data.rename(index=str, columns={"0": "LL", "1": "pval"})
    data.LL = round(data.LL,3)
    data.pval = round(data.pval,3)
    data = data[~data["word"].isin(ss[0:int(stopwords_1.value)])]
    

    return data

# set up widgets

#stats_para = Paragraph(text="""Summary statistics for rank and frequency distributions of the texts:""",
#width=300)
#stats = PreText(text='', width=300)

ticker1 = Select(value='Huckleberry_Finn', options=nix('A_Farewell_to_Arms', DEFAULT_TICKERS))
ticker2 = Select(value='A_Farewell_to_Arms', options=nix('Huckleberry_Finn', DEFAULT_TICKERS))

stopwords_1 = Select(title="Remove most frequent words:", value="0", options=["0","10", "20", "50", "100", "200"])
#stopwords_2 = Slider(title="Remove most frequent words", start=0, end=200, value=0, step=1)
#vis_menu = Select(value='linear',options=nix('log',DEFAULT_VIS))

#df_1=get_data('HUCK','FAREWELL')
#set up joint df

source = ColumnDataSource(data=dict(word=[],rank_x=[],rank_y=[],freq_x=[],freq_y=[],
				    sum_x=[],sum_y=[],rel_x=[],rel_y=[],rel_diff=[],LL=[],pval=[]))

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
    <b>rel_diff: </b> @rel_diff <br>
    <b>log_l: </b> @LL <br>
    <b>p_val: </b> @pval <br>
"""


TOOLS = "pan,wheel_zoom,box_select,reset,hover"
#TOOLS = "box_select,help,pan,wheel_zoom,box_zoom,reset,previewsave"
#TOOLS = [BoxSelectTool(),PanTool(),custom_hover,WheelZoomTool(),SaveTool(),ResetTool()]
#TOOLS_r = [BoxSelectTool(),PanTool(),custom_hover_right,WheelZoomTool(),SaveTool(),ResetTool()]

# create a new plot and add a renderer
left_lin = figure(tools=TOOLS,x_axis_type='log', y_axis_type='log',
		  plot_width=450, plot_height=450, output_backend="webgl")
left_lin.circle('rank_x', 'rel_x', source=source,alpha=0.6, size=6,selection_color="orange", hover_color="firebrick")
#label = Label(x=1.1, y=18, text="FUGG", text_font_size='30pt', text_color='#eeeeee')
#left.add_layout(label)
hoverL = left_lin.select(dict(type=HoverTool))
hoverL.tooltips={"word": "@word","rank":"@rank_x","freq":"@freq_x","per_10k":"@rel_x","LL":"@LL","pval":"@pval"}
#panel_llin = Panel(child=left_lin, title='linear')


#left_log = figure(tools=TOOLS,x_axis_type='log', y_axis_type='log',
#		  plot_width=350, plot_height=350)#, output_backend="webgl")
#left_log.circle('rank_x', 'rel_x', source=source,alpha=0.6, size=6,selection_color="orange", hover_color="firebrick")
#label = Label(x=1.1, y=18, text="FUGG", text_font_size='30pt', text_color='#eeeeee')
#left.add_layout(label)
#hover = left_log.select(dict(type=HoverTool))
#hover.tooltips={"word": "@word","rank":"@rank_x","freq":"@freq_x","per_10k":"@rel_x","LL":"@LL","pval":"@pval"}
#panel_llog = Panel(child=left_log, title='log')


# create another new plot and add a renderer
right_lin = figure(tools=TOOLS,x_axis_type='log',y_axis_type='log',
		   plot_width=450, plot_height=450, output_backend="webgl")
right_lin.circle('rank_y', 'rel_y', source=source,alpha=0.6, size=6,selection_color="orange", hover_color="firebrick")
hoverR = right_lin.select(dict(type=HoverTool))
hoverR.tooltips={"word": "@word","rank":"@rank_y","freq":"@freq_y","per_10k":"@rel_y","LL":"@LL","pval":"@pval"}
#right.add_tools(custom_hover)
#panel_rlin = Panel(child=right_lin, title='linear')


#right_log = figure(tools=TOOLS,x_axis_type='log',y_axis_type='log',
#		   plot_width=350, plot_height=350)#, output_backend="webgl")
#right_log.circle('rank_y', 'rel_y', source=source,alpha=0.6, size=6,selection_color="orange", hover_color="firebrick")
#hover = right_log.select(dict(type=HoverTool))
#hover.tooltips={"word": "@word","rank":"@rank_y","freq":"@freq_y","per_10k":"@rel_y","LL":"@LL","pval":"@pval"}
#right.add_tools(custom_hover)
#panel_rlog = Panel(child=right_log, title='log')


#tabs_l = Tabs(tabs=[panel_llin,panel_llog])
#tabs_r = Tabs(tabs=[panel_rlin,panel_rlog])

columns_l = [
    TableColumn(field="rank_x", title="rank"),
    TableColumn(field="freq_x", title="freq"),
    TableColumn(field="word", title="word"),
    TableColumn(field="rel_diff", title="rel_diff"),
    TableColumn(field="LL", title="Log-likelihood")
]
columns_r = [
    TableColumn(field="rank_y", title="rank"),
    TableColumn(field="freq_y", title="freq"),
    TableColumn(field="word", title="word"),
    TableColumn(field="rel_diff", title="rel_diff"),
    TableColumn(field="LL", title="Log-likelihood")
]
top10_l = DataTable(source=source, columns=columns_l,width=450, height=450)
top10_r = DataTable(source=source, columns=columns_r,width=450, height=450)
#data[['rank_x','rank_y','freq_x','freq_y']]

p = gridplot([[left_lin, right_lin],[top10_l,top10_r]],
                        toolbar_location = "above",
                        toolbar_options=dict(logo=None),
			#sizing_mode='stretch_both'
                        )
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
    #data = data[~data["word"].isin(ss[0:stopwords_1.value])]
    source.data = source.from_df(data[['word','rank_x','rank_y','freq_x','freq_y','sum_x','sum_y','rel_x','rel_y','rel_diff','LL','pval']])
    #source.data = dict(data)
    #source_static.data = source.data
    #update_stats(data, t1, t2)
    left_lin.title.text = '%s, Gini coef. = %s' % (t1, round(gini(source.data["freq_x"][source.data["freq_x"]>0]),3))
    #left_log.title.text = '%s, Gini coef. = %s' % (t1, round(gini(source.data["freq_x"][source.data["freq_x"]>0]),3))
    right_lin.title.text = '%s, Gini coef. = %s' % (t2, round(gini(source.data["freq_y"][source.data["freq_y"]>0]),3))
    #right_log.title.text = '%s, Gini coef. = %s' % (t2, round(gini(source.data["freq_y"][source.data["freq_y"]>0]),3))
    #ts1.title.text, ts2.title.text = t1, t2

#def update_stats(data, t1, t2):
#    stats.text = str(round(data[['rank_x','rank_y','freq_x','freq_y']].describe(),2))

#def update_top10(data,t1,t2):

ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)

def selection_change(attrname, old, new):
    t1, t2 = ticker1.value, ticker2.value
    data = get_data(t1, t2)
    selected = source.selected['1d']['indices']
    if selected:
        data = data.iloc[selected, :]
    #update()
    #update_stats(data, t1, t2)

#control1 = stopwords_1
stopwords_1.on_change('value', lambda attr, old, new: update())
#control2 = stopwords_2
#control2.on_change('value', lambda attr, old, new: update())


source.on_change('selected', selection_change)

# set up layout
MODE = 'fixed'

#col1=column(tabs_l,top10_l)
#col2=column(tabs_r,top10_r)
widgets = widgetbox(ticker1, ticker2, stopwords_1, width=150)
#widgets.sizing_mode = 'scale_width'
main_row = row(p,widgets)
#main_row.sizing_mode = 'fixed'
#series = column(ts1, ts2)
layout = column(main_row)

#layout=layout([[p],[ticker1,ticker2,stats]])
#p = gridplot([[tabs_l, tabs_r, widgetbox],[top10_l,top10_r, None]])
# initialize

update()

curdoc().add_root(layout)
curdoc().title = "ZipfExplorer"
