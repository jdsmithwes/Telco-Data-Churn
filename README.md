
<a href="https://colab.research.google.com/github/jdsmithwes/Telco-Data-Churn/blob/master/Module3_project_master.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Telco Churn Project
Jamaal Smith 





## Business Case Intro

This analysis will seek to examine whether or not one can predict a customer's decision to remain with a phone company or leave the company. This analysis is important to businesses that must allocate resources to maximize their revenue. Additionally, besides losing paying customers in the near term, companies often have to spend more money to attract new customers than they spend retaining existing customers. 

While there are many factors and data points that can be collected on customers, the primary goal of this analysis is to identify the handful of feature variables that a company can monitor to take preventive measures to mitigate customer churn.

### Additional Context (From Context)

Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

Companies usually make a distinction between voluntary churn and involuntary churn. Voluntary churn occurs due to a decision by the customer to switch to another company or service provider, involuntary churn occurs due to circumstances such as a customer's relocation to a long-term care facility, death, or the relocation to a distant location. In most applications, involuntary reasons for churn are excluded from the analytical models. Analysts tend to concentrate on voluntary churn, because it typically occurs due to factors of the company-customer relationship which companies control, such as how billing interactions are handled or how after-sales help is provided.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
```

## Data Analysis

### First Impressions of Data


```python
#import the data
df = pd.read_csv('churn_data.csv')
from scrubbing import info
#from scrubbing import null
info(df)
```

    The dataframe has a shape of:
    
     (3333, 21)
    The dataframe has the following datatypes:
    
     state                      object
    account length              int64
    area code                   int64
    phone number               object
    international plan         object
    voice mail plan            object
    number vmail messages       int64
    total day minutes         float64
    total day calls             int64
    total day charge          float64
    total eve minutes         float64
    total eve calls             int64
    total eve charge          float64
    total night minutes       float64
    total night calls           int64
    total night charge        float64
    total intl minutes        float64
    total intl calls            int64
    total intl charge         float64
    customer service calls      int64
    churn                        bool
    dtype: object
    The total number of each datatype is:
    
    
     float64    8
    int64      8
    object     4
    bool       1
    dtype: int64





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>account length</th>
      <th>area code</th>
      <th>phone number</th>
      <th>international plan</th>
      <th>voice mail plan</th>
      <th>number vmail messages</th>
      <th>total day minutes</th>
      <th>total day calls</th>
      <th>total day charge</th>
      <th>...</th>
      <th>total eve calls</th>
      <th>total eve charge</th>
      <th>total night minutes</th>
      <th>total night calls</th>
      <th>total night charge</th>
      <th>total intl minutes</th>
      <th>total intl calls</th>
      <th>total intl charge</th>
      <th>customer service calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.isnull().sum()
```




    state                     0
    account length            0
    area code                 0
    phone number              0
    international plan        0
    voice mail plan           0
    number vmail messages     0
    total day minutes         0
    total day calls           0
    total day charge          0
    total eve minutes         0
    total eve calls           0
    total eve charge          0
    total night minutes       0
    total night calls         0
    total night charge        0
    total intl minutes        0
    total intl calls          0
    total intl charge         0
    customer service calls    0
    churn                     0
    dtype: int64



Based on initial review of the data, we can conclude that the dataset is comprised of 21 columns and 3333 rows. Further, there are no null values in the data set.

The next step in preparing the data for the model is to convert the object type in the phone number, international plan, and voicemail plan columns for the machine learning model.

For now, I will only focus on the international and voicemail plan columns. I made this decision because the phone number a person is assigned plays no role in whether or not a person keeps their service.

### Phone Number & State Column Treatment

### EDA


```python
#Frequency of Churn Phenomenon
sns.countplot(x='churn',data=df)
plt.title('Frequency of Churn')
```




    Text(0.5, 1.0, 'Frequency of Churn')




![png](output_14_1.png)



```python
a = (df['churn'].sum())
b = (len(df['churn']))
percent_churn = a/b
percent_remain = 1-percent_churn

print('The percentage of Churn in dataset is:', percent_churn*100)
print('The percentage of remaining in dataset is:', percent_remain*100)
```

    The percentage of Churn in dataset is: 14.491449144914492
    The percentage of remaining in dataset is: 85.5085508550855


#### Feature Variable Examination

Because the churn variable is categorical, many visual tools will be limited and not fruitful. As a proxy, comparisons of feature variables against the account length column might be useful. It is worth examining whether certain behaviors by customers result in longer account lengths.


```python
#for visualization below
df.columns
```




    Index(['state', 'account length', 'area code', 'phone number',
           'international plan', 'voice mail plan', 'number vmail messages',
           'total day minutes', 'total day calls', 'total day charge',
           'total eve minutes', 'total eve calls', 'total eve charge',
           'total night minutes', 'total night calls', 'total night charge',
           'total intl minutes', 'total intl calls', 'total intl charge',
           'customer service calls', 'churn'],
          dtype='object')




```python
#jointplot with account length as proxy for churn
#fig,ax = plt. subplots()
fig = plt.figure(figsize=(14,4))
features = ['number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'customer service calls',]
#for i in range(2):
    #for j in range(7):
    
for feature in features:
    ax = sns.jointplot(feature,'account length',data=df,kind='reg')
    plt.show()
```


    <Figure size 1008x288 with 0 Axes>



![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)



![png](output_19_7.png)



![png](output_19_8.png)



![png](output_19_9.png)



![png](output_19_10.png)



![png](output_19_11.png)



![png](output_19_12.png)



![png](output_19_13.png)



![png](output_19_14.png)



```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(df[features],df['account length'])
coef = lr.coef_
df_coef = pd.DataFrame(coef,columns=['correlation'],index=features)
df_coef

plt.figure(figsize=(25,8))
sns.barplot(x=df_coef.index,y=df_coef['correlation'], data=df_coef)

#df_coef.columns = ['Feature','Correlation']
#df_coef.sort_values(by='')
print(df_coef)
```

                            correlation
    number vmail messages     -0.014050
    total day minutes         14.795190
    total day calls            0.074535
    total day charge         -87.004538
    total eve minutes        -20.910801
    total eve calls            0.036977
    total eve charge         245.958743
    total night minutes        2.723135
    total night calls         -0.025976
    total night charge       -60.673414
    total intl minutes       -25.959918
    total intl calls           0.318870
    total intl charge         96.559527
    customer service calls    -0.088122



![png](output_20_1.png)



```python
#boxplot with account length as proxy for churn
#fig,ax = plt. subplots()
fig = plt.figure(figsize=(14,4))
features = ['number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'customer service calls',]
#for i in range(2):
    #for j in range(7):
   
for feature in features:
    ax = sns.boxplot(df[feature],data=df)
    plt.show()
```


![png](output_21_0.png)



![png](output_21_1.png)



![png](output_21_2.png)



![png](output_21_3.png)



![png](output_21_4.png)



![png](output_21_5.png)



![png](output_21_6.png)



![png](output_21_7.png)



![png](output_21_8.png)



![png](output_21_9.png)



![png](output_21_10.png)



![png](output_21_11.png)



![png](output_21_12.png)



![png](output_21_13.png)


The above visualizations do not show any significant relationships between the selected feature variables and proxy for churn, account length. Based on the human eye, the regression line did seem to be positively sloped for some of the daytime characteristics.

The phone number column can be excluded from the data set. However, when examing churn on a state-by-state basis, there is enough variance in these figures to suggest that state could be a feature variable that might be worth considering for inclusion in the model.

### Analysis of State Feature


```python
#groupby method used to get churn numbers by state
df_state = df.groupby('state',axis=0).sum()
state_churn = df_state['churn']
states = df['state'].unique()
sorted_states = sorted(states)
```


```python
#import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



```python
#parameters for graph
data = dict(type='choropleth',
            colorscale = 'ylorbr',
            locations = sorted_states,
            z = df_state['churn'],
            locationmode = 'USA-states',
            text = sorted_states,
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Churn amount"}
            ) 
```


```python
#paramenters for graph
layout = dict(title = 'Churn by State',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )
```


```python
choromap = go.Figure(data = [data],layout = layout)
```


```python
fig = plt.figure(figsize=(20,10))
iplot(choromap)
```


<div>
        
        
            <div id="365f5a92-05d6-4c89-8ebc-f38103410663" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("365f5a92-05d6-4c89-8ebc-f38103410663")) {
                    Plotly.newPlot(
                        '365f5a92-05d6-4c89-8ebc-f38103410663',
                        [{"colorbar": {"title": {"text": "Churn amount"}}, "colorscale": [[0.0, "rgb(255,255,229)"], [0.125, "rgb(255,247,188)"], [0.25, "rgb(254,227,145)"], [0.375, "rgb(254,196,79)"], [0.5, "rgb(254,153,41)"], [0.625, "rgb(236,112,20)"], [0.75, "rgb(204,76,2)"], [0.875, "rgb(153,52,4)"], [1.0, "rgb(102,37,6)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "marker": {"line": {"color": "rgb(255,255,255)", "width": 2}}, "text": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "type": "choropleth", "z": [3.0, 8.0, 11.0, 4.0, 9.0, 9.0, 12.0, 5.0, 9.0, 8.0, 8.0, 3.0, 3.0, 9.0, 5.0, 9.0, 13.0, 8.0, 4.0, 11.0, 17.0, 13.0, 16.0, 15.0, 7.0, 14.0, 14.0, 11.0, 6.0, 5.0, 9.0, 18.0, 6.0, 14.0, 15.0, 10.0, 9.0, 11.0, 8.0, 6.0, 14.0, 8.0, 5.0, 18.0, 10.0, 5.0, 8.0, 14.0, 7.0, 10.0, 9.0]}],
                        {"geo": {"lakecolor": "rgb(85,173,240)", "scope": "usa", "showlakes": true}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Churn by State"}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('365f5a92-05d6-4c89-8ebc-f38103410663');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



    <Figure size 1440x720 with 0 Axes>


Based on the choropleth map displayed above, the state feature variable is worth keeping in the final analysis.

## Data Preprocessing

### Addressing Categorical Variables


```python
df_encode = df


category = [key for key in dict(df_encode.dtypes) if dict(df_encode.dtypes)[key] in ['bool','object']]

LE = LabelEncoder()
for i in category:
    LE.fit(df[i])
    df_encode[i] = LE.transform(df_encode[i])
df_encode.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>account length</th>
      <th>area code</th>
      <th>phone number</th>
      <th>international plan</th>
      <th>voice mail plan</th>
      <th>number vmail messages</th>
      <th>total day minutes</th>
      <th>total day calls</th>
      <th>total day charge</th>
      <th>...</th>
      <th>total eve calls</th>
      <th>total eve charge</th>
      <th>total night minutes</th>
      <th>total night calls</th>
      <th>total night charge</th>
      <th>total intl minutes</th>
      <th>total intl calls</th>
      <th>total intl charge</th>
      <th>customer service calls</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>16</td>
      <td>128</td>
      <td>415</td>
      <td>1926</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>...</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>35</td>
      <td>107</td>
      <td>415</td>
      <td>1575</td>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>...</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>31</td>
      <td>137</td>
      <td>415</td>
      <td>1117</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>...</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>35</td>
      <td>84</td>
      <td>408</td>
      <td>1707</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>...</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>36</td>
      <td>75</td>
      <td>415</td>
      <td>110</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>...</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Outlier Treatment


```python
#treatment of outliers based on boxplot visualization above
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(df_encode)
df_encode = transformer.transform(df_encode)
df_encode = pd.DataFrame(df_encode,columns = df.columns)
df_encode.head(5)
print(df_encode.shape)
```

    (3333, 21)


### Assigning Variables


```python
y = df_encode['churn']
X = df_encode.drop(['churn','account length'],axis=1)
```

For now, the states column will be left in the dataset and later feature selection measures will provide a statistically significant method for determining whether or not this explanatory variable will be included in the final model.

### Data Split for Cross-Validation


```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,train_size=.8)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
```

### Feature Scaling


```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train,columns = X.columns)
X_test = pd.DataFrame(X_test,columns= X.columns)
```

# Machine Learning Models

## XGBoost Classifier


```python
# train the model

model = XGBClassifier()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train,y_train,eval_metric=['error','logloss'],eval_set=eval_set)
```

    [0]	validation_0-error:0.095649	validation_0-logloss:0.627421	validation_1-error:0.089955	validation_1-logloss:0.625042
    [1]	validation_0-error:0.095649	validation_0-logloss:0.573636	validation_1-error:0.089955	validation_1-logloss:0.569213
    [2]	validation_0-error:0.095649	validation_0-logloss:0.528397	validation_1-error:0.089955	validation_1-logloss:0.522644
    [3]	validation_0-error:0.095649	validation_0-logloss:0.490886	validation_1-error:0.088456	validation_1-logloss:0.484135
    [4]	validation_0-error:0.095649	validation_0-logloss:0.457544	validation_1-error:0.089955	validation_1-logloss:0.4504
    [5]	validation_0-error:0.095274	validation_0-logloss:0.429131	validation_1-error:0.085457	validation_1-logloss:0.420606
    [6]	validation_0-error:0.094149	validation_0-logloss:0.40259	validation_1-error:0.086957	validation_1-logloss:0.394448
    [7]	validation_0-error:0.093023	validation_0-logloss:0.380024	validation_1-error:0.085457	validation_1-logloss:0.371574
    [8]	validation_0-error:0.091148	validation_0-logloss:0.360476	validation_1-error:0.083958	validation_1-logloss:0.351746
    [9]	validation_0-error:0.088147	validation_0-logloss:0.343038	validation_1-error:0.083958	validation_1-logloss:0.334738


    /opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/preprocessing/label.py:219: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    
    /opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/preprocessing/label.py:252: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    


    [10]	validation_0-error:0.087022	validation_0-logloss:0.327738	validation_1-error:0.082459	validation_1-logloss:0.319572
    [11]	validation_0-error:0.088522	validation_0-logloss:0.314712	validation_1-error:0.083958	validation_1-logloss:0.307221
    [12]	validation_0-error:0.086647	validation_0-logloss:0.303031	validation_1-error:0.082459	validation_1-logloss:0.295687
    [13]	validation_0-error:0.083271	validation_0-logloss:0.292476	validation_1-error:0.082459	validation_1-logloss:0.286249
    [14]	validation_0-error:0.076519	validation_0-logloss:0.283228	validation_1-error:0.08096	validation_1-logloss:0.276513
    [15]	validation_0-error:0.077269	validation_0-logloss:0.275229	validation_1-error:0.08096	validation_1-logloss:0.26923
    [16]	validation_0-error:0.075019	validation_0-logloss:0.267819	validation_1-error:0.08096	validation_1-logloss:0.262442
    [17]	validation_0-error:0.074269	validation_0-logloss:0.259608	validation_1-error:0.08096	validation_1-logloss:0.254678
    [18]	validation_0-error:0.073518	validation_0-logloss:0.252697	validation_1-error:0.085457	validation_1-logloss:0.248103
    [19]	validation_0-error:0.070893	validation_0-logloss:0.246991	validation_1-error:0.08096	validation_1-logloss:0.242808
    [20]	validation_0-error:0.062641	validation_0-logloss:0.238693	validation_1-error:0.074963	validation_1-logloss:0.235136
    [21]	validation_0-error:0.05964	validation_0-logloss:0.233947	validation_1-error:0.070465	validation_1-logloss:0.230794
    [22]	validation_0-error:0.055889	validation_0-logloss:0.22953	validation_1-error:0.073463	validation_1-logloss:0.226767
    [23]	validation_0-error:0.057014	validation_0-logloss:0.223344	validation_1-error:0.070465	validation_1-logloss:0.221096
    [24]	validation_0-error:0.048762	validation_0-logloss:0.218137	validation_1-error:0.05997	validation_1-logloss:0.216136
    [25]	validation_0-error:0.047262	validation_0-logloss:0.214429	validation_1-error:0.056972	validation_1-logloss:0.212784
    [26]	validation_0-error:0.057389	validation_0-logloss:0.212075	validation_1-error:0.067466	validation_1-logloss:0.210266
    [27]	validation_0-error:0.055889	validation_0-logloss:0.208982	validation_1-error:0.068966	validation_1-logloss:0.208102
    [28]	validation_0-error:0.055139	validation_0-logloss:0.206292	validation_1-error:0.068966	validation_1-logloss:0.205213
    [29]	validation_0-error:0.054014	validation_0-logloss:0.203508	validation_1-error:0.067466	validation_1-logloss:0.202368
    [30]	validation_0-error:0.055514	validation_0-logloss:0.20092	validation_1-error:0.067466	validation_1-logloss:0.199986
    [31]	validation_0-error:0.051013	validation_0-logloss:0.198853	validation_1-error:0.064468	validation_1-logloss:0.198285
    [32]	validation_0-error:0.053263	validation_0-logloss:0.196879	validation_1-error:0.065967	validation_1-logloss:0.1975
    [33]	validation_0-error:0.051388	validation_0-logloss:0.195417	validation_1-error:0.064468	validation_1-logloss:0.196053
    [34]	validation_0-error:0.051013	validation_0-logloss:0.194052	validation_1-error:0.064468	validation_1-logloss:0.194767
    [35]	validation_0-error:0.052888	validation_0-logloss:0.192953	validation_1-error:0.062969	validation_1-logloss:0.193745
    [36]	validation_0-error:0.049512	validation_0-logloss:0.191128	validation_1-error:0.056972	validation_1-logloss:0.192332
    [37]	validation_0-error:0.050263	validation_0-logloss:0.189951	validation_1-error:0.05997	validation_1-logloss:0.191168
    [38]	validation_0-error:0.049512	validation_0-logloss:0.188833	validation_1-error:0.058471	validation_1-logloss:0.189841
    [39]	validation_0-error:0.050638	validation_0-logloss:0.187669	validation_1-error:0.056972	validation_1-logloss:0.18847
    [40]	validation_0-error:0.051013	validation_0-logloss:0.186973	validation_1-error:0.058471	validation_1-logloss:0.18823
    [41]	validation_0-error:0.050638	validation_0-logloss:0.186143	validation_1-error:0.055472	validation_1-logloss:0.187823
    [42]	validation_0-error:0.051013	validation_0-logloss:0.185194	validation_1-error:0.056972	validation_1-logloss:0.186773
    [43]	validation_0-error:0.050638	validation_0-logloss:0.184309	validation_1-error:0.056972	validation_1-logloss:0.186415
    [44]	validation_0-error:0.051013	validation_0-logloss:0.183554	validation_1-error:0.055472	validation_1-logloss:0.186129
    [45]	validation_0-error:0.049887	validation_0-logloss:0.182668	validation_1-error:0.053973	validation_1-logloss:0.185649
    [46]	validation_0-error:0.050263	validation_0-logloss:0.182034	validation_1-error:0.053973	validation_1-logloss:0.185493
    [47]	validation_0-error:0.050263	validation_0-logloss:0.181347	validation_1-error:0.055472	validation_1-logloss:0.185028
    [48]	validation_0-error:0.045761	validation_0-logloss:0.179542	validation_1-error:0.050975	validation_1-logloss:0.183452
    [49]	validation_0-error:0.045011	validation_0-logloss:0.178484	validation_1-error:0.050975	validation_1-logloss:0.182699
    [50]	validation_0-error:0.045386	validation_0-logloss:0.177299	validation_1-error:0.050975	validation_1-logloss:0.182616
    [51]	validation_0-error:0.044636	validation_0-logloss:0.17621	validation_1-error:0.050975	validation_1-logloss:0.182571
    [52]	validation_0-error:0.044636	validation_0-logloss:0.175737	validation_1-error:0.050975	validation_1-logloss:0.182308
    [53]	validation_0-error:0.045011	validation_0-logloss:0.174727	validation_1-error:0.053973	validation_1-logloss:0.182213
    [54]	validation_0-error:0.045386	validation_0-logloss:0.17441	validation_1-error:0.053973	validation_1-logloss:0.181999
    [55]	validation_0-error:0.044636	validation_0-logloss:0.173485	validation_1-error:0.052474	validation_1-logloss:0.181752
    [56]	validation_0-error:0.040885	validation_0-logloss:0.171951	validation_1-error:0.049475	validation_1-logloss:0.180106
    [57]	validation_0-error:0.04051	validation_0-logloss:0.171258	validation_1-error:0.049475	validation_1-logloss:0.179681
    [58]	validation_0-error:0.03976	validation_0-logloss:0.170683	validation_1-error:0.046477	validation_1-logloss:0.179621
    [59]	validation_0-error:0.039385	validation_0-logloss:0.169799	validation_1-error:0.047976	validation_1-logloss:0.179325
    [60]	validation_0-error:0.03901	validation_0-logloss:0.169417	validation_1-error:0.047976	validation_1-logloss:0.179117
    [61]	validation_0-error:0.038635	validation_0-logloss:0.168832	validation_1-error:0.047976	validation_1-logloss:0.178443
    [62]	validation_0-error:0.03826	validation_0-logloss:0.167992	validation_1-error:0.046477	validation_1-logloss:0.17832
    [63]	validation_0-error:0.038635	validation_0-logloss:0.167646	validation_1-error:0.044978	validation_1-logloss:0.178208
    [64]	validation_0-error:0.03826	validation_0-logloss:0.167228	validation_1-error:0.044978	validation_1-logloss:0.178172
    [65]	validation_0-error:0.037884	validation_0-logloss:0.16659	validation_1-error:0.044978	validation_1-logloss:0.177924
    [66]	validation_0-error:0.037134	validation_0-logloss:0.165209	validation_1-error:0.046477	validation_1-logloss:0.178009
    [67]	validation_0-error:0.037134	validation_0-logloss:0.164883	validation_1-error:0.046477	validation_1-logloss:0.177784
    [68]	validation_0-error:0.037134	validation_0-logloss:0.164543	validation_1-error:0.046477	validation_1-logloss:0.177799
    [69]	validation_0-error:0.036759	validation_0-logloss:0.163984	validation_1-error:0.044978	validation_1-logloss:0.177863
    [70]	validation_0-error:0.036759	validation_0-logloss:0.163318	validation_1-error:0.044978	validation_1-logloss:0.177904
    [71]	validation_0-error:0.036009	validation_0-logloss:0.162744	validation_1-error:0.044978	validation_1-logloss:0.177912
    [72]	validation_0-error:0.036759	validation_0-logloss:0.162236	validation_1-error:0.044978	validation_1-logloss:0.177138
    [73]	validation_0-error:0.035634	validation_0-logloss:0.160622	validation_1-error:0.046477	validation_1-logloss:0.175806
    [74]	validation_0-error:0.035634	validation_0-logloss:0.160185	validation_1-error:0.046477	validation_1-logloss:0.175601
    [75]	validation_0-error:0.035259	validation_0-logloss:0.159752	validation_1-error:0.046477	validation_1-logloss:0.174963
    [76]	validation_0-error:0.035259	validation_0-logloss:0.159416	validation_1-error:0.047976	validation_1-logloss:0.175093
    [77]	validation_0-error:0.035259	validation_0-logloss:0.158185	validation_1-error:0.046477	validation_1-logloss:0.17548
    [78]	validation_0-error:0.035259	validation_0-logloss:0.157968	validation_1-error:0.046477	validation_1-logloss:0.175388
    [79]	validation_0-error:0.034509	validation_0-logloss:0.157268	validation_1-error:0.046477	validation_1-logloss:0.175156
    [80]	validation_0-error:0.034134	validation_0-logloss:0.15663	validation_1-error:0.046477	validation_1-logloss:0.175043
    [81]	validation_0-error:0.034134	validation_0-logloss:0.156118	validation_1-error:0.046477	validation_1-logloss:0.175113
    [82]	validation_0-error:0.034509	validation_0-logloss:0.155713	validation_1-error:0.046477	validation_1-logloss:0.175169
    [83]	validation_0-error:0.034884	validation_0-logloss:0.155343	validation_1-error:0.044978	validation_1-logloss:0.175021
    [84]	validation_0-error:0.034509	validation_0-logloss:0.154699	validation_1-error:0.046477	validation_1-logloss:0.17527
    [85]	validation_0-error:0.034509	validation_0-logloss:0.154325	validation_1-error:0.047976	validation_1-logloss:0.174655
    [86]	validation_0-error:0.034509	validation_0-logloss:0.154023	validation_1-error:0.044978	validation_1-logloss:0.174849
    [87]	validation_0-error:0.034134	validation_0-logloss:0.153549	validation_1-error:0.044978	validation_1-logloss:0.174863
    [88]	validation_0-error:0.034134	validation_0-logloss:0.153096	validation_1-error:0.044978	validation_1-logloss:0.174369
    [89]	validation_0-error:0.034134	validation_0-logloss:0.152741	validation_1-error:0.044978	validation_1-logloss:0.174466
    [90]	validation_0-error:0.033758	validation_0-logloss:0.151688	validation_1-error:0.043478	validation_1-logloss:0.174599
    [91]	validation_0-error:0.034509	validation_0-logloss:0.151043	validation_1-error:0.043478	validation_1-logloss:0.174275
    [92]	validation_0-error:0.034509	validation_0-logloss:0.150813	validation_1-error:0.044978	validation_1-logloss:0.174278
    [93]	validation_0-error:0.034509	validation_0-logloss:0.150647	validation_1-error:0.044978	validation_1-logloss:0.174231
    [94]	validation_0-error:0.033758	validation_0-logloss:0.150011	validation_1-error:0.044978	validation_1-logloss:0.174098
    [95]	validation_0-error:0.033383	validation_0-logloss:0.149707	validation_1-error:0.044978	validation_1-logloss:0.174094
    [96]	validation_0-error:0.033383	validation_0-logloss:0.149209	validation_1-error:0.044978	validation_1-logloss:0.174253
    [97]	validation_0-error:0.033383	validation_0-logloss:0.148685	validation_1-error:0.044978	validation_1-logloss:0.173955
    [98]	validation_0-error:0.033383	validation_0-logloss:0.147805	validation_1-error:0.046477	validation_1-logloss:0.173875
    [99]	validation_0-error:0.033383	validation_0-logloss:0.147434	validation_1-error:0.046477	validation_1-logloss:0.17394





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)




```python
#predict
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print( "The model's accuracy is",accuracy)
```

    The model's accuracy is 0.9535232383808095



```python
from sklearn.metrics import classification_report
classification_report(y_pred,y_test)
```




    '              precision    recall  f1-score   support\n\n         0.0       0.99      0.96      0.97       595\n         1.0       0.72      0.93      0.81        72\n\n    accuracy                           0.95       667\n   macro avg       0.86      0.94      0.89       667\nweighted avg       0.96      0.95      0.96       667\n'




```python
#confusion matrix
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

print('The Accuracy Score for this model is {acc}'.format(acc=acc))
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 10)
plt.tight_layout()
annot_kws = {"ha": 'left',"va": 'top'}
sns.heatmap(cm, annot=True,annot_kws=annot_kws, fmt="d", linewidths=.2, square = True,); 


#ax = sns.heatmap(data, annot=True, annot_kws= annot_kws)
```

    The Accuracy Score for this model is 0.9535232383808095



![png](output_49_1.png)



```python
# make predictions for test data
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# retrieve performance metrics
results = model.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

from matplotlib import pyplot
# plot log loss
fig, ax = pyplot.subplots(figsize=(6,6))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Epochs/#Trees')
pyplot.title('XGBoost Log Loss')
pyplot.show()
```

    Accuracy: 95.35%



![png](output_50_1.png)



```python
# plot classification error
fig, ax = pyplot.subplots(figsize=(6,6))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Epochs')
pyplot.title('XGBoost Classification Error')
pyplot.show()
```


![png](output_51_0.png)



```python
#feature importance - weight
plot_importance(model,max_num_features=20)
plt.title("Feature Importance - Weight")
plt.show()
```


![png](output_52_0.png)



```python
#feature importance - cover
plot_importance(model,max_num_features=20,importance_type='cover')
plt.title("Feature Importance - Cover")
plt.show()
```


![png](output_53_0.png)



```python
#feature importance - gain
plot_importance(model,max_num_features=20,importance_type='gain')
plt.title("Feature Importance - Gain")
plt.show()
```


![png](output_54_0.png)



```python
from regression import preprocess
from regression import optimal_alpha
optimal_alpha(X_train,y_train)
```

    Optimal Alpha Value: 0



![png](output_55_1.png)



```python
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.9535232383808095





    Text(0.5, 1, 'Confusion Matrix')




![png](output_56_2.png)



```python
# Feature Selection with SelectFromModel

from sklearn.feature_selection import SelectFromModel
thresholds = sorted(model.feature_importances_)


output_dict = {}

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=0.10, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
     
```

    Thresh=0.000, n=4, Accuracy: 90.40%
    Thresh=0.000, n=4, Accuracy: 90.40%
    Thresh=0.000, n=4, Accuracy: 90.40%
    Thresh=0.000, n=4, Accuracy: 90.40%
    Thresh=0.015, n=4, Accuracy: 90.40%
    Thresh=0.019, n=4, Accuracy: 90.40%
    Thresh=0.020, n=4, Accuracy: 90.40%
    Thresh=0.022, n=4, Accuracy: 90.40%
    Thresh=0.023, n=4, Accuracy: 90.40%
    Thresh=0.029, n=4, Accuracy: 90.40%
    Thresh=0.035, n=4, Accuracy: 90.40%
    Thresh=0.040, n=4, Accuracy: 90.40%
    Thresh=0.052, n=4, Accuracy: 90.40%
    Thresh=0.061, n=4, Accuracy: 90.40%
    Thresh=0.084, n=4, Accuracy: 90.40%
    Thresh=0.126, n=4, Accuracy: 90.40%
    Thresh=0.134, n=4, Accuracy: 90.40%
    Thresh=0.158, n=4, Accuracy: 90.40%
    Thresh=0.183, n=4, Accuracy: 90.40%


## XGBoost with Feature Selection


```python
results=pd.DataFrame()
results['columns']=X_train.columns
results['importances'] = model.feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)

plt.figure(figsize=(40,8))
plt.tight_layout()
sns.barplot(x=results['columns'],y=results['importances'], data=results)

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a25215c18>




![png](output_59_1.png)



```python
df_encode.columns
```




    Index(['state', 'account length', 'area code', 'phone number',
           'international plan', 'voice mail plan', 'number vmail messages',
           'total day minutes', 'total day calls', 'total day charge',
           'total eve minutes', 'total eve calls', 'total eve charge',
           'total night minutes', 'total night calls', 'total night charge',
           'total intl minutes', 'total intl calls', 'total intl charge',
           'customer service calls', 'churn'],
          dtype='object')




```python
#selecting top features based on importance
X_new = df_encode[['international plan','customer service calls','total day minutes','voice mail plan']]
y_new = df_encode['churn']
```


```python
#data split
X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(X_new,y_new)
```


```python
# train the model

model_new = XGBClassifier()
eval_set = [(X_train_new, y_train_new), (X_test_new, y_test_new)]
model_new.fit(X_train_new,y_train_new,eval_metric=['error','logloss'],eval_set=eval_set)
```

    [0]	validation_0-error:0.096038	validation_0-logloss:0.627248	validation_1-error:0.097122	validation_1-logloss:0.628098
    [1]	validation_0-error:0.096038	validation_0-logloss:0.57331	validation_1-error:0.097122	validation_1-logloss:0.574833
    [2]	validation_0-error:0.096038	validation_0-logloss:0.528517	validation_1-error:0.097122	validation_1-logloss:0.530488
    [3]	validation_0-error:0.096038	validation_0-logloss:0.49093	validation_1-error:0.097122	validation_1-logloss:0.493442
    [4]	validation_0-error:0.096038	validation_0-logloss:0.458429	validation_1-error:0.097122	validation_1-logloss:0.460918
    [5]	validation_0-error:0.096038	validation_0-logloss:0.430744	validation_1-error:0.097122	validation_1-logloss:0.433411
    [6]	validation_0-error:0.096038	validation_0-logloss:0.406978	validation_1-error:0.097122	validation_1-logloss:0.409631
    [7]	validation_0-error:0.096038	validation_0-logloss:0.385869	validation_1-error:0.097122	validation_1-logloss:0.388571
    [8]	validation_0-error:0.096038	validation_0-logloss:0.367593	validation_1-error:0.097122	validation_1-logloss:0.370532
    [9]	validation_0-error:0.096038	validation_0-logloss:0.351678	validation_1-error:0.097122	validation_1-logloss:0.354678
    [10]	validation_0-error:0.096038	validation_0-logloss:0.337851	validation_1-error:0.097122	validation_1-logloss:0.340836
    [11]	validation_0-error:0.096038	validation_0-logloss:0.325777	validation_1-error:0.097122	validation_1-logloss:0.328991
    [12]	validation_0-error:0.096038	validation_0-logloss:0.315212	validation_1-error:0.097122	validation_1-logloss:0.318568
    [13]	validation_0-error:0.096038	validation_0-logloss:0.305931	validation_1-error:0.097122	validation_1-logloss:0.309378
    [14]	validation_0-error:0.096038	validation_0-logloss:0.297806	validation_1-error:0.097122	validation_1-logloss:0.301389
    [15]	validation_0-error:0.096439	validation_0-logloss:0.290702	validation_1-error:0.098321	validation_1-logloss:0.294708
    [16]	validation_0-error:0.096439	validation_0-logloss:0.284449	validation_1-error:0.098321	validation_1-logloss:0.288477
    [17]	validation_0-error:0.096439	validation_0-logloss:0.279077	validation_1-error:0.098321	validation_1-logloss:0.283606
    [18]	validation_0-error:0.096439	validation_0-logloss:0.274286	validation_1-error:0.098321	validation_1-logloss:0.278905
    [19]	validation_0-error:0.096439	validation_0-logloss:0.270361	validation_1-error:0.098321	validation_1-logloss:0.275575
    [20]	validation_0-error:0.096439	validation_0-logloss:0.266607	validation_1-error:0.098321	validation_1-logloss:0.271986
    [21]	validation_0-error:0.096439	validation_0-logloss:0.263317	validation_1-error:0.098321	validation_1-logloss:0.268782
    [22]	validation_0-error:0.096839	validation_0-logloss:0.260562	validation_1-error:0.098321	validation_1-logloss:0.266115
    [23]	validation_0-error:0.096839	validation_0-logloss:0.25798	validation_1-error:0.098321	validation_1-logloss:0.26392
    [24]	validation_0-error:0.096439	validation_0-logloss:0.255941	validation_1-error:0.098321	validation_1-logloss:0.262223
    [25]	validation_0-error:0.096439	validation_0-logloss:0.253769	validation_1-error:0.098321	validation_1-logloss:0.260443
    [26]	validation_0-error:0.096839	validation_0-logloss:0.251962	validation_1-error:0.098321	validation_1-logloss:0.258772
    [27]	validation_0-error:0.096839	validation_0-logloss:0.250443	validation_1-error:0.098321	validation_1-logloss:0.257534
    [28]	validation_0-error:0.096839	validation_0-logloss:0.248994	validation_1-error:0.098321	validation_1-logloss:0.256427
    [29]	validation_0-error:0.096839	validation_0-logloss:0.247624	validation_1-error:0.098321	validation_1-logloss:0.255441
    [30]	validation_0-error:0.096839	validation_0-logloss:0.246361	validation_1-error:0.098321	validation_1-logloss:0.254527
    [31]	validation_0-error:0.096839	validation_0-logloss:0.245266	validation_1-error:0.098321	validation_1-logloss:0.25358
    [32]	validation_0-error:0.096038	validation_0-logloss:0.24436	validation_1-error:0.098321	validation_1-logloss:0.252985
    [33]	validation_0-error:0.095638	validation_0-logloss:0.243447	validation_1-error:0.098321	validation_1-logloss:0.252413
    [34]	validation_0-error:0.095638	validation_0-logloss:0.242578	validation_1-error:0.098321	validation_1-logloss:0.251696
    [35]	validation_0-error:0.095638	validation_0-logloss:0.242093	validation_1-error:0.098321	validation_1-logloss:0.251554
    [36]	validation_0-error:0.095638	validation_0-logloss:0.24169	validation_1-error:0.098321	validation_1-logloss:0.251398
    [37]	validation_0-error:0.095238	validation_0-logloss:0.240934	validation_1-error:0.098321	validation_1-logloss:0.251218
    [38]	validation_0-error:0.095638	validation_0-logloss:0.240611	validation_1-error:0.095923	validation_1-logloss:0.250966
    [39]	validation_0-error:0.095638	validation_0-logloss:0.240157	validation_1-error:0.095923	validation_1-logloss:0.251023
    [40]	validation_0-error:0.095638	validation_0-logloss:0.239878	validation_1-error:0.095923	validation_1-logloss:0.250914
    [41]	validation_0-error:0.095638	validation_0-logloss:0.239629	validation_1-error:0.095923	validation_1-logloss:0.250734
    [42]	validation_0-error:0.095638	validation_0-logloss:0.239397	validation_1-error:0.095923	validation_1-logloss:0.250617
    [43]	validation_0-error:0.095638	validation_0-logloss:0.239186	validation_1-error:0.095923	validation_1-logloss:0.250649
    [44]	validation_0-error:0.095638	validation_0-logloss:0.238987	validation_1-error:0.095923	validation_1-logloss:0.250724
    [45]	validation_0-error:0.095638	validation_0-logloss:0.238429	validation_1-error:0.095923	validation_1-logloss:0.250598
    [46]	validation_0-error:0.095638	validation_0-logloss:0.238268	validation_1-error:0.095923	validation_1-logloss:0.250732
    [47]	validation_0-error:0.095238	validation_0-logloss:0.237916	validation_1-error:0.095923	validation_1-logloss:0.250893
    [48]	validation_0-error:0.094838	validation_0-logloss:0.237437	validation_1-error:0.095923	validation_1-logloss:0.250775
    [49]	validation_0-error:0.094838	validation_0-logloss:0.237297	validation_1-error:0.095923	validation_1-logloss:0.250919
    [50]	validation_0-error:0.094838	validation_0-logloss:0.237017	validation_1-error:0.095923	validation_1-logloss:0.25078
    [51]	validation_0-error:0.094838	validation_0-logloss:0.236506	validation_1-error:0.095923	validation_1-logloss:0.250828
    [52]	validation_0-error:0.094838	validation_0-logloss:0.236207	validation_1-error:0.095923	validation_1-logloss:0.250772
    [53]	validation_0-error:0.094838	validation_0-logloss:0.235783	validation_1-error:0.095923	validation_1-logloss:0.250984
    [54]	validation_0-error:0.094838	validation_0-logloss:0.235157	validation_1-error:0.095923	validation_1-logloss:0.250912
    [55]	validation_0-error:0.094038	validation_0-logloss:0.23471	validation_1-error:0.097122	validation_1-logloss:0.250999
    [56]	validation_0-error:0.093637	validation_0-logloss:0.234284	validation_1-error:0.097122	validation_1-logloss:0.250964
    [57]	validation_0-error:0.093637	validation_0-logloss:0.234028	validation_1-error:0.097122	validation_1-logloss:0.251012
    [58]	validation_0-error:0.093637	validation_0-logloss:0.233796	validation_1-error:0.097122	validation_1-logloss:0.251103
    [59]	validation_0-error:0.094438	validation_0-logloss:0.233578	validation_1-error:0.095923	validation_1-logloss:0.251167
    [60]	validation_0-error:0.094438	validation_0-logloss:0.233378	validation_1-error:0.095923	validation_1-logloss:0.251272
    [61]	validation_0-error:0.094438	validation_0-logloss:0.233079	validation_1-error:0.095923	validation_1-logloss:0.251373
    [62]	validation_0-error:0.094038	validation_0-logloss:0.232821	validation_1-error:0.095923	validation_1-logloss:0.251605
    [63]	validation_0-error:0.092837	validation_0-logloss:0.232587	validation_1-error:0.098321	validation_1-logloss:0.251696
    [64]	validation_0-error:0.092437	validation_0-logloss:0.232123	validation_1-error:0.097122	validation_1-logloss:0.251723
    [65]	validation_0-error:0.090436	validation_0-logloss:0.23178	validation_1-error:0.097122	validation_1-logloss:0.251736
    [66]	validation_0-error:0.090836	validation_0-logloss:0.231438	validation_1-error:0.098321	validation_1-logloss:0.25188
    [67]	validation_0-error:0.090436	validation_0-logloss:0.231132	validation_1-error:0.097122	validation_1-logloss:0.251892
    [68]	validation_0-error:0.090436	validation_0-logloss:0.230956	validation_1-error:0.097122	validation_1-logloss:0.251781
    [69]	validation_0-error:0.090436	validation_0-logloss:0.230624	validation_1-error:0.097122	validation_1-logloss:0.251806
    [70]	validation_0-error:0.090036	validation_0-logloss:0.230202	validation_1-error:0.097122	validation_1-logloss:0.25179
    [71]	validation_0-error:0.090036	validation_0-logloss:0.230108	validation_1-error:0.097122	validation_1-logloss:0.251831
    [72]	validation_0-error:0.090036	validation_0-logloss:0.229932	validation_1-error:0.097122	validation_1-logloss:0.251918
    [73]	validation_0-error:0.090436	validation_0-logloss:0.229777	validation_1-error:0.097122	validation_1-logloss:0.252065
    [74]	validation_0-error:0.090036	validation_0-logloss:0.229493	validation_1-error:0.097122	validation_1-logloss:0.252118
    [75]	validation_0-error:0.090036	validation_0-logloss:0.2292	validation_1-error:0.097122	validation_1-logloss:0.252183
    [76]	validation_0-error:0.090036	validation_0-logloss:0.229062	validation_1-error:0.097122	validation_1-logloss:0.252413
    [77]	validation_0-error:0.090036	validation_0-logloss:0.228656	validation_1-error:0.097122	validation_1-logloss:0.252491
    [78]	validation_0-error:0.089636	validation_0-logloss:0.228406	validation_1-error:0.09952	validation_1-logloss:0.252563
    [79]	validation_0-error:0.090036	validation_0-logloss:0.228027	validation_1-error:0.097122	validation_1-logloss:0.252344
    [80]	validation_0-error:0.090836	validation_0-logloss:0.227641	validation_1-error:0.098321	validation_1-logloss:0.252473
    [81]	validation_0-error:0.090836	validation_0-logloss:0.227538	validation_1-error:0.098321	validation_1-logloss:0.252514
    [82]	validation_0-error:0.090436	validation_0-logloss:0.227286	validation_1-error:0.101918	validation_1-logloss:0.252667
    [83]	validation_0-error:0.090436	validation_0-logloss:0.227089	validation_1-error:0.101918	validation_1-logloss:0.25263
    [84]	validation_0-error:0.090436	validation_0-logloss:0.226855	validation_1-error:0.101918	validation_1-logloss:0.252781
    [85]	validation_0-error:0.090436	validation_0-logloss:0.226763	validation_1-error:0.101918	validation_1-logloss:0.252827
    [86]	validation_0-error:0.088836	validation_0-logloss:0.226559	validation_1-error:0.101918	validation_1-logloss:0.25292
    [87]	validation_0-error:0.088836	validation_0-logloss:0.226347	validation_1-error:0.101918	validation_1-logloss:0.253172
    [88]	validation_0-error:0.088836	validation_0-logloss:0.226169	validation_1-error:0.101918	validation_1-logloss:0.25327
    [89]	validation_0-error:0.088836	validation_0-logloss:0.225989	validation_1-error:0.103118	validation_1-logloss:0.253374
    [90]	validation_0-error:0.087635	validation_0-logloss:0.225647	validation_1-error:0.101918	validation_1-logloss:0.253381
    [91]	validation_0-error:0.087635	validation_0-logloss:0.22532	validation_1-error:0.101918	validation_1-logloss:0.253396
    [92]	validation_0-error:0.087635	validation_0-logloss:0.224865	validation_1-error:0.101918	validation_1-logloss:0.253483
    [93]	validation_0-error:0.088035	validation_0-logloss:0.224584	validation_1-error:0.103118	validation_1-logloss:0.253593
    [94]	validation_0-error:0.088035	validation_0-logloss:0.224471	validation_1-error:0.103118	validation_1-logloss:0.25374
    [95]	validation_0-error:0.088035	validation_0-logloss:0.224173	validation_1-error:0.103118	validation_1-logloss:0.253775
    [96]	validation_0-error:0.087635	validation_0-logloss:0.224017	validation_1-error:0.100719	validation_1-logloss:0.253845
    [97]	validation_0-error:0.088035	validation_0-logloss:0.223903	validation_1-error:0.101918	validation_1-logloss:0.253842
    [98]	validation_0-error:0.088035	validation_0-logloss:0.223637	validation_1-error:0.101918	validation_1-logloss:0.253874
    [99]	validation_0-error:0.088035	validation_0-logloss:0.223348	validation_1-error:0.101918	validation_1-logloss:0.253895





    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)




```python
# make predictions for test data
y_pred_new = model_new.predict(X_test_new)
predictions = [round(value) for value in y_pred_new]

# evaluate predictions
accuracy = accuracy_score(y_test_new, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# retrieve performance metrics
results_new = model_new.evals_result()
epochs = len(results_new['validation_0']['error'])
x_axis = range(0, epochs)

from matplotlib import pyplot
# plot log loss
fig, ax = pyplot.subplots(figsize=(6,6))
ax.plot(x_axis, results_new['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results_new['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.xlabel('Epochs/#Trees')
pyplot.title('XGBoost with SelectFromModel Log Loss')
pyplot.show()
```

    Accuracy: 89.81%



![png](output_64_1.png)



```python
# plot classification error
fig, ax = pyplot.subplots(figsize=(6,6))
ax.plot(x_axis, results_new['validation_0']['error'], label='Train')
ax.plot(x_axis, results_new['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.xlabel('Epochs')
pyplot.title('XGBoost with SelectFromModel Classification Error')
pyplot.show()
```


![png](output_65_0.png)



```python
cm = confusion_matrix(y_test_new,y_pred_new)
acc = accuracy_score(y_test_new,y_pred_new)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.8980815347721822





    Text(0.5, 1, 'Confusion Matrix')




![png](output_66_2.png)


## Logistic Model


```python
#Logistic Model with GridCV to find best parameters for model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

lr = LogisticRegression()
param_grid = {'class_weight' : ['balanced', None], 
              'penalty' : ['l2','l1'], 
              'C' : [0.001, 0.01, 0.1, 1, 10, 100],
              'solver': ['saga','liblinear']}
grid = GridSearchCV(estimator = lr, cv=5, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)
grid.fit(X_train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_

```

    Fitting 5 folds for each of 48 candidates, totalling 240 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  65 tasks      | elapsed:    4.1s


    Best Score:0.8069969993944408
    Best Parameters: {'C': 10, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'saga'}


    [Parallel(n_jobs=-1)]: Done 240 out of 240 | elapsed:   20.6s finished



```python
#predictions
lr = LogisticRegression(**best_parameters)
lr.fit(X_train,y_train)
y_pred_log = lr.predict(X_test)
accuracy_score(y_test,y_pred_log)
```




    0.7976011994002998




```python
#confusion matrix visualization
cm = confusion_matrix(y_test,y_pred_log)
acc = accuracy_score(y_test,y_pred_log)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.7976011994002998





    Text(0.5, 1, 'Confusion Matrix')




![png](output_70_2.png)



```python
#getting feature names after RFE
from sklearn.feature_selection import RFE
cols = list(X_train.columns)
selector = RFE(lr,4)
X_rfe = selector.fit(X_train,y_train)

features = pd.Series(selector.support_,index = cols)
selected_features_rfe = features[features==True].index

X_log_select = df_encode[selected_features_rfe]
print(selected_features_rfe)
```

    Index(['international plan', 'voice mail plan', 'total day minutes',
           'customer service calls'],
          dtype='object')



```python
#splitting data
X_train_log_select, X_test_log_select, y_train_log_select, y_test_log_select = train_test_split(X_log_select,y)

```


```python
#train model
lr.fit(X_train_log_select,y_train_log_select)

#prediction
y_pred_log_select = lr.predict(X_test_log_select)
```


```python
#confusion matrix visualization
cm = confusion_matrix(y_test_log_select,y_pred_log_select)
acc = accuracy_score(y_test_log_select,y_pred_log_select)
print('The Accuracy Score for this model is {acc}'.format(acc=acc))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = False, cmap = 'Reds');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#plt.text(verticalalignment='center')
all_sample_title = 'Accuracy Score: {0}'.format(acc)
plt.title('Confusion Matrix', size = 15)
```

    The Accuracy Score for this model is 0.749400479616307





    Text(0.5, 1, 'Confusion Matrix')




![png](output_74_2.png)

## Recommendation
As the feature selection algorithm highlighted, the four key feature variables that the telephone company must monitor are: international plan, voicemail plan, total day minutes and customer service calls.

The company should examine states with the highest level of churn to understand whether their churn is the result of other market forces such as rampant competition or whether these states might not have the same infrastructure investment as sttes with lower churn levels.


