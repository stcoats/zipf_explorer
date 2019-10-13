from os.path import dirname, join
import pandas as pd
import powerlaw
import scipy.stats
from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot, widgetbox,Spacer
from bokeh.models.widgets import FileInput, Slider, DataTable, TableColumn
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, CDSView, BooleanFilter,PanTool,WheelZoomTool,SaveTool,ResetTool,Label
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import PreText, Select, Tabs, Panel, Paragraph
from bokeh.plotting import figure
import numpy as np
import chardet
from scipy.stats import chi2_contingency
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import nltk
import regex
from collections import OrderedDict
from io import StringIO, BytesIO
import base64

nltk.download('punkt')
stopw_file = join(dirname(__file__),'data', 'google_200words.txt')
words=open(stopw_file).read()
ss=words.split()

DEFAULT_TICKERS=  ['Brown_TOTAL', 'Brown_adventure', 'Brown_belles_lettres',
		   'Brown_editorial', 'Brown_fiction', 'Brown_government',
		   'Brown_hobbies', 'Brown_humor', 'Brown_learned', 'Brown_lore',
		   'Brown_mystery', 'Brown_news', 'Brown_religion', 'Brown_reviews',
		   'Brown_romance', 'Brown_science_fiction', 'FROWN_TOTAL', 'Frown_TOTAL',
		   'aesop-fables', 'anonymous-book_mormon', 'austen-emma',
		   'austen-persuasion', 'austen-sense', 'bible-kjv', 'blake-poems',
		   'bryant-stories', 'burgess-busterbrown', 'carroll-alice',
		   'chesterton-ball', 'chesterton-brown', 'chesterton-thursday',
		   'conan_doyle-return_sherlock', 'dickens-christmas_carol', 
		   'edgeworth-parents', 'hamilton_jay_madison-fed',
		   'hardy-return_of_the_native', 'hemingway-farewell',
		   'inaugural_addresses', 'james-the_american', 'james-the_europeans',
		   'london-call_of_the_wild', 'marx_engels-communist',
		   'melville-moby_dick', 'milton-paradise', 'plato-republic',
		   'shakespeare-caesar', 'shakespeare-hamlet', 'shakespeare-macbeth',
		   'shelley-frankenstein', 'sinclair-the_jungle', 'sophocles-oedipus',
		   'stowe-uncle_toms_cabin', 'twain-huckleberry', 'twain-tom_sawyer',
		   'wells-war_of_the_worlds', 'whitman-leaves', 'wilde-dorian_gray',
		   'User_input']

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
#def load_ticker(ticker):
#    fname = join(dirname(__file__),'data', '%s.csv' % ticker)
#    data = pd.read_csv(fname,index_col=0)
#    #data=data.drop('na', axis=1)
#    data["rel"]=data["freq"]*10000/data["freq"].sum()
#    data["rank"]=data.index
#    #data = data.set_index('date')
#    #return pd.DataFrame({ticker: data.c, ticker+'_returns': data.c.diff()})
#    return data

def to_freq_list(text):
    words = word_tokenize(text)
    dist = FreqDist([x.lower() for x in words if regex.search("[\p{Letter}0-9]",x)])
    vv=OrderedDict(sorted(dist.items(), key=lambda x:x[1]) )
    vv1=OrderedDict(reversed(list(vv.items())))
    df = pd.DataFrame(list(vv1.items()))
    df.index += 1
    df.columns = ['word','freq']
    return df
def get_data(t1, t2):
    if ticker1.value == 'User_input':
      csv = base64.b64decode(user1.value)
      #text = textract.process(user1.value) #if user1.value.endswith(tuple(ext)) else base64.b64decode(user1.value) 
      enc = chardet.detect(csv)
      #print(text.decode(enc["encoding"]))
      df1 = to_freq_list(csv.decode(enc["encoding"],"ignore"))
      df1["rel"]=df1["freq"]*10000/df1["freq"].sum()
      df1["rank"]=df1.index
    else:
      df1name = join(dirname(__file__),'data', '%s.csv' % ticker1.value)
      df1 = pd.read_csv(df1name,index_col=0)
      df1["rel"]=df1["freq"]*10000/df1["freq"].sum()
      df1["rank"]=df1.index
    if ticker2.value == 'User_input':
      csv = base64.b64decode(user2.value)
      #df2 = pd.read_csv(BytesIO(csv))
      enc = chardet.detect(csv)
      df2 = to_freq_list(csv.decode(enc["encoding"],"ignore"))
      df2["rel"]=df2["freq"]*10000/df2["freq"].sum()
      df2["rank"]=df2.index
    else:
      df2name = join(dirname(__file__),'data', '%s.csv' % ticker2.value)
      df2 = pd.read_csv(df2name,index_col=0)
      df2["rel"]=df2["freq"]*10000/df2["freq"].sum()
      df2["rank"]=df2.index
    #print(len(df1),len(df2))
    data = pd.merge(df1, df2, on='word',how='inner').fillna(0)
    data = data.dropna()
    #data = data["rel_x" > 0
    data["rel_diff"]=round(data["rel_x"]-data["rel_y"],3)
    data["sum_x"]=data["freq_x"].sum()
    data["sum_y"]=data["freq_y"].sum()
    data=pd.concat([data,pd.DataFrame(data.apply(chisq,axis=1))],axis=1)
    data.columns = data.columns.astype(str)
    data=data.rename(index=str, columns={"0": "LL", "1": "pval"})
    data.LL = round(data.LL,3)
    data.pval = round(data.pval,3)
    data = data[~data["word"].isin(ss[0:int(stopwords_1.value)])]
    return data,df1,df2
    
ticker1 = Select(value='twain-huckleberry', options=nix('hemingway-farewell', DEFAULT_TICKERS))
ticker2 = Select(value='hemingway-farewell', options=nix('twain-huckleberry', DEFAULT_TICKERS))
user1 = FileInput(accept='.csv,.txt,.doc,.docx')
user2 = FileInput(accept='.csv,.txt,.doc,.docx')

stopwords_1 = Select(title="Remove most frequent words:", value="0", options=["0","10", "20", "50", "100", "200"])

source = ColumnDataSource(data=dict(word=[],rank_x=[],rank_y=[],freq_x=[],freq_y=[],
		    sum_x=[],sum_y=[],rel_x=[],rel_y=[],rel_diff=[],LL=[],pval=[]))

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


TOOLS = "pan,box_zoom,wheel_zoom,box_select,reset,hover"

left_lin = figure(tools=TOOLS,x_axis_type='linear', y_axis_type='linear', plot_width=400, plot_height=400, sizing_mode='fixed',
	    output_backend="webgl")
left_lin.circle('rank_x', 'rel_x', source=source,alpha=0.6, size=10,selection_color="red", hover_color="red")
hoverL = left_lin.select(dict(type=HoverTool))
hoverL.tooltips={"word": "@word","rank":"@rank_x","freq":"@freq_x","per_10k":"@rel_x","LL":"@LL","pval":"@pval"}
panel_llin = Panel(child=left_lin, title='linear')

left_log = figure(tools=TOOLS,x_axis_type='log', y_axis_type='log', plot_width=400, plot_height=400, sizing_mode='fixed',
	    output_backend="webgl")
left_log.circle('rank_x', 'rel_x', source=source,alpha=0.6, size=10,selection_color="red", hover_color="red")
hoverL = left_log.select(dict(type=HoverTool))
hoverL.tooltips={"word": "@word","rank":"@rank_x","freq":"@freq_x","per_10k":"@rel_x","LL":"@LL","pval":"@pval"}
panel_llog = Panel(child=left_log, title='log')

right_lin = figure(tools=TOOLS,x_axis_type='linear',y_axis_type='linear', plot_width=400, plot_height=400, sizing_mode='fixed',
	    output_backend="webgl")
right_lin.circle('rank_y', 'rel_y', source=source,alpha=0.6, size=10,selection_color="red", hover_color="red")
hoverR = right_lin.select(dict(type=HoverTool))
hoverR.tooltips={"word": "@word","rank":"@rank_y","freq":"@freq_y","per_10k":"@rel_y","LL":"@LL","pval":"@pval"}
panel_rlin = Panel(child=right_lin, title='linear')

right_log = figure(tools=TOOLS,x_axis_type='log',y_axis_type='log', plot_width=400, plot_height=400, sizing_mode='fixed',
	    output_backend="webgl")
right_log.circle('rank_y', 'rel_y', source=source,alpha=0.6, size=10,selection_color="red", hover_color="red")
hoverR = right_log.select(dict(type=HoverTool))
hoverR.tooltips={"word": "@word","rank":"@rank_y","freq":"@freq_y","per_10k":"@rel_y","LL":"@LL","pval":"@pval"}
panel_rlog = Panel(child=right_log, title='log')

tabs_l = Tabs(tabs=[panel_llin,panel_llog])
tabs_r = Tabs(tabs=[panel_rlin,panel_rlog])

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
top10_l = DataTable(source=source, columns=columns_l,width=400, height=350)
top10_r = DataTable(source=source, columns=columns_r,width=400, height=350)
#data[['rank_x','rank_y','freq_x','freq_y']]

p = gridplot(children=[[tabs_l,Spacer(width=10), tabs_r],[top10_l,Spacer(width=10),top10_r]],
			toolbar_location = "above",
			toolbar_options=dict(logo=None),
	    #sizing_mode='fixed',

			)

# set up callbacks

def ticker1_change(attrname, old, new):
    #ticker2.options = nix(new, DEFAULT_TICKERS)
    update()

def ticker2_change(attrname, old, new):
    #ticker1.options = nix(new, DEFAULT_TICKERS)
    update()
    
def user1_change(attrname, old, new):
    ticker1.value = "User_input"
    #ticker2.options = nix(new, DEFAULT_TICKERS)
    update()

def user2_change(attrname, old, new):
    ticker2.value = "User_input"
    #ticker1.options = nix(new, DEFAULT_TICKERS)
    update()
dict1 = {
'x':[0]*6,
'y':[1,1,1,2,2,2]
	    }

table_source = ColumnDataSource(data=dict1)

def update(selected=None):
    t1, t2 = ticker1.value, ticker2.value
    data = get_data(t1, t2)
    update_stats(data, data[1], data[2])
    source.data = source.from_df(data[0])
    #selection_1 = np.array(load_ticker(t1)[~load_ticker(t1)["word"].isin(ss[0:int(stopwords_1.value)])]["freq"].astype(float))
    selection_1 = np.array(data[0][~data[0]["word"].isin(ss[0:int(stopwords_1.value)])]["freq_x"].astype(float))
    #selection_2 = np.array(load_ticker(t2)[~load_ticker(t2)["word"].isin(ss[0:int(stopwords_1.value)])]["freq"].astype(float))
    selection_2 = np.array(data[0][~data[0]["word"].isin(ss[0:int(stopwords_1.value)])]["freq_y"].astype(float))
    left_lin.title.text = '%s, TTR = %s' % (t1, round(len(selection_1)/sum(selection_1),3)) + ', Gini = %s' % round(gini(selection_1),3) + ', ⍺ = %s' % str(round(powerlaw.Fit(selection_1).alpha,3)) + ', 𝑯 = %s' % str(round(ent(pd.Series(selection_1)),3))
    left_log.title.text = '%s, TTR = %s' % (t1, round(len(selection_1)/sum(selection_1),3)) + ', Gini = %s' % round(gini(selection_1),3) + ', ⍺ = %s' % str(round(powerlaw.Fit(selection_1).alpha,3)) + ', 𝑯 = %s' % str(round(ent(pd.Series(selection_1)),3))
    right_lin.title.text = '%s, TTR = %s' % (t2, round(len(selection_2)/sum(selection_2),3)) + ', Gini = %s' % round(gini(selection_2),3) + ', ⍺ = %s' % str(round(powerlaw.Fit(selection_2).alpha,3)) + ', 𝑯 = %s' % str(round(ent(pd.Series(selection_2)),3))
    right_log.title.text = '%s, TTR = %s' % (t2, round(len(selection_2)/sum(selection_2),3)) + ', Gini = %s' % round(gini(selection_2),3) + ', ⍺ = %s' % str(round(powerlaw.Fit(selection_2).alpha,3)) + ', 𝑯 = %s' % str(round(ent(pd.Series(selection_2)),3))
    left_lin.title.text_font_size = '7pt'
    left_log.title.text_font_size = '7pt'
    right_lin.title.text_font_size = '7pt'
    right_log.title.text_font_size = '7pt'
    right_lin.x_range = left_lin.x_range
    right_lin.y_range = left_lin.y_range
    right_log.x_range = left_log.x_range
    right_log.y_range = left_log.y_range
    # table_source.data = table_source.from_df(df)


ticker1.on_change('value', ticker1_change)
ticker2.on_change('value', ticker2_change)
user1.on_change('value',user1_change)
user2.on_change('value',user2_change)

def selection_change(attrname, old, new):
    t1, t2 = ticker1.value, ticker2.value
    data = get_data(t1, t2)
    
    selected = source.selected['1d']['indices']
    #update_stats(data)
    table_source.data = table_source.from_df(data)
    if selected:
      data = data.iloc[selected, :]
    
stats = PreText(text="hi", width=300)
#stats.on_change('value', stats)

def update_stats(data,df1,df2):
    shared_types = data[0][~data[0]["word"].isin(ss[0:int(stopwords_1.value)])]
    t1_types = data[1][~data[1]["word"].isin(ss[0:int(stopwords_1.value)])]
    t2_types = data[2][~data[2]["word"].isin(ss[0:int(stopwords_1.value)])]
    stats.text = str("t1: {} types, {} tokens\nt2: {} types, {} tokens\n{}% of t1 types in t2\n{}% of t2 types in t1".
		     format(len(t1_types),
			    t1_types["freq"].sum(),
			    len(t2_types),
			    t2_types["freq"].sum(),
			  round((len(shared_types)/len(t1_types)*100),2),
			round((len(shared_types)/len(t2_types)*100),2)))
stopwords_1.on_change('value', lambda attr, old, new: update())

source.on_change('selected', selection_change)

MODE = 'fixed'

widgets = widgetbox(ticker1, user1, ticker2, user2, stopwords_1, stats, width=250)
main_row = row(p,widgets)
sec_row = row(top10_r,Spacer(width=40),top10_r)
layout = column(main_row)

update()

curdoc().add_root(layout)
curdoc().title = "ZipfExplorer"

