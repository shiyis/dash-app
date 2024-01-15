
import dash
import pandas as pd
import os
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, callback
import pandas as pd
import numpy as np
import os
import numpy as np
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse


states = pd.read_csv("./data/states.csv")
candidates = pd.read_csv("./data/2022/processed_weball.csv")


dash.register_page(__name__, title='Text-based Ideal Points',location='sidebar')


PAGE_STYLE = {
    "position":"absolute",
    "margin":"4.5rem 4rem 0rem 22rem",
    "color":"#000",
    "text-shadow":"#000 0 0",
    'whiteSpace': 'pre-wrap'
}



custom = './data/custom_data/clean/author_map.txt'
if not os.path.isfile(custom):
    dataPath = './data/2022/candidate-tweets-2020/clean/'
else:
    dataPath = '/'.join(custom.split('/')[:-1])
# Load data
author_indices = np.load(dataPath + "author_indices.npy")

counts = sparse.load_npz(dataPath + "counts.npz")

with open(dataPath + "vocabulary.txt",'r') as f:
    vocabulary = f.readlines()

with open(dataPath + "author_map.txt",'r') as f:
    author_map = f.readlines()

author_map = np.array(author_map)
num_authors = int(author_indices.max() + 1)
num_documents, num_words = counts.shape
pre_initialize_parameters = True

neutral_topic_mean = np.load("./data/2022/neutral_topic_mean.npy")
negative_topic_mean = np.load("./data/2022/neutral_topic_mean.npy")
positive_topic_mean = np.load("./data/2022/positive_topic_mean.npy")
authors = pd.read_csv("./data/2022/authors.csv")
authors["name"] = authors["name"].str.replace("\n", "")



layout = html.Div([html.H5("Text-based Ideal Point Generated for Author's Political Leaning"),
html.Hr(),
html.P([dcc.Markdown("""To continue with our project objective of measuring and investigating how a politician's internal motivation aligns with their external actions (disbursements and expenditures with respect to their political acitivities and agenda; which we have disclosed the most basic information in the previous section, or [C4FE Exploratory Data Analysis](https://c4fe-tbip.onrender.com/page1), this section will present a statistical topic modeling over the authors/politicians' tweets."""),html.A("TBIP", href="https://www.aclweb.org/anthology/2020.acl-main.475/"), """ is an unsupervised probabilistic topic model (Keyon V., Suresh N., David B. et al.) evaluates texts to quantify the political stances of their authors. The model does not require any text labeled with an ideology, nor does it use political parties or votes.""", html.P(""""""),dcc.Markdown("""Instead, it assesses the `latent political viewpoints` of text writers and how `per-topic word choice` varies according to the author's political stance `("ideological topics")` given a corpus of political text and the author of each document."""),
"""Below are trained results for the list of 2022 federal election candidates' ideal points and topic aggregation of their Twitter archive.""",
dbc.Row(
    [dbc.Col(dbc.Row(dbc.Col(children=[html.Label(['Select State'], style={'font-size': '13px', "text-align": "left", "color": "#808080"}), dcc.Dropdown(pd.DataFrame(pd.read_csv("./data/states.csv"))['name'].tolist(),id='state-dropdown-p2')],id='states-col-p2'))),
    dbc.Col(dbc.Row(dbc.Col(children=[html.Label(['Select Candidate'], style={'font-size': '13px', "text-align": "left", "off-set":4, "color": "#808080"}), dcc.Dropdown(id='names-dropdown-p2')],id='cand-names-col-p2'),id='cand-names-row-p2')),

    ]),

dbc.Row([
dbc.Col([
    dcc.Graph(id='bar-graph-plotly', figure={})
]),
dbc.Col(children=[
    dag.AgGrid(
        id='grid',
        rowData=authors.to_dict("records"),
        columnDefs=[{"field": i} for i in authors.columns],
        columnSize="sizeToFit",
        style={"text-align": "center"}
    )
], md=20, style={"margin": "2rem 0 0 0"}),
], className='mt-4',style={"text-align": "center"}),
html.P(""),
html.Br(),
html.H5("Topic Modeling with Variational Encoding"),
html.Hr(),
"""The model performs inference using """,html.A("variational inference.", href="https://arxiv.org/abs/1601.00670"),""" with """, html.A("reparameterization", href="https://arxiv.org/abs/1312.6114"),html.A(" gradients.", href="https://arxiv.org/abs/1401.4082"), 
""" What this means in plain language is that imagine you have a bunch of data, like pictures of cats. Each cat picture can be described by a set of features—things like the color of the fur, the size of the ears, and the length of the tail. Now, let's say you want to understand the hidden or latent factors that contribute to these features. The challenge is that there might be some randomness or uncertainty in these latent factors.""",dbc.Row(children=[dbc.Col(children=["""
1. Encoding the Data:""",html.Code("""
The variational encoder takes each cat picture and encodes it into a set of numbers hat represent the key features. 
However, instead of giving you a single set of numbers. It provides a distribution. This distribution tells you how likely different values of these features are.""")]),
dbc.Col(children=["""                                                                                                                                              
2. Introducing Randomness:""",html.Code("""
To account for uncertainty in the latent factors, the encoder introduces a bit of randomness into the encoding process. 
This means that even for the same cat picture, you might get slightly different sets of numbers each time due to this randomness.""")]),dbc.Col(children=["""
3. Measuring Latent Distribution:""",html.Code("""
The distribution you get from the variational encoder reflects the uncertainty or variability in the latent factors. It's like saying, "well, we're not exactly sure about the size of the ears, but here's a range of possiblities, and here's how likely each possibility is."
""")]),dbc.Col(children=["""
4. Sampling from the Distribution:""",html.Code("""
Now, when generating new cat pictures, we sample from this distribution. Sampling means picking a specific set of numbers from the distribution. Since the distribution captures the uncertainty, each sample gives a slightly different version of the cat picture, exploring the potential variations.""")])]),"""
This is an extension of another popular algorithm the Latent Dirichlet Allocation (LDA). In the context of textual topic modeling, variational inference helps approximate the posterior distribution of latent variables, such as the distribution of topics in documents and words in topics. In variational inference, we need to specify a family of distributions from which we will choose an approximation to the true (but often intractable) posterior distribution. This family of distributions is called the "variational family." Common choices for the variational family include mean-field variational families.""",

"""

In summary, variational inference, with the help of a variational family, allows us to approximate complex posterior distributions in topic modeling. It helps uncover latent topics and their distributions in a collection of documents, providing valuable insights into the underlying thematic structures.""", html.P(""""""),
dcc.Markdown("""Now we want to see above descriptions materialized in actual formulas. Again, because it is intractable to evaluate the posterior distribution $p(\\theta, \\beta, \\eta, x | y)$, so the posterior is estimated with a distribution $q_\\phi(\\theta, \\beta,\\eta,x)$, parameterized by $\\phi$ through minimizing the KL-Divergence between $q$ and the posterior (put simple is the distance between these two distributions), which is equivalent to maximizing the ELBO (or the Evidence Lower Bound):""", mathjax=True),dcc.Markdown("""
        $$\\mathbb{L}_{\\theta,\phi}(\\mathbf{x})=\mathbb{E}_{q_{\\phi}(\\mathbf{z}|\\mathbf{x})}[\\log p_{\\theta}(\\mathbf{x},\\mathbf{z})-\\log q_{\\phi}(\\mathbf{z}|\\mathbf{x})]$$""",mathjax=True,style={"text-align": "center"}), dcc.Markdown(                    
"""The variational family is set to be the mean-field family, meaning the latent variables factorize over documents $d$, topics $k$, and authors $$s$$:""", mathjax=True),
dbc.Row([dcc.Markdown(
"""$$q_\\phi(\\theta, \\beta, \\eta, x) = \\prod_{d,k,s} q(\\theta_d)q(\\beta_k)q(\\eta_k)q(x_s).$$""",mathjax=True)], style={"text-align": "center"}),
dcc.Markdown("""
Lognormal factors are used for the positive variables and Gaussian factors for the real variables:
"""),
dbc.Row(
    [dcc.Markdown("""
        $$q(\\theta_d) = \\text{LogNormal}_K(\\mu_{\\theta_d}\\sigma^2_{\\theta_d})$$

        $$q(\\beta_k) = \\text{LogNormal}_V(\\mu_{\\beta_k}, \\sigma^2_{\\beta_k})$$

        $$q(\\eta_k) = \\mathcal{N}_V(\\mu_{\\eta_k}, \\sigma^2_{\\eta_k})$$

        $$q(x_s) = \\mathcal{N}(\\mu_{x_s}, \\sigma^2_{x_s}).$$""",mathjax=True)], style={"text-align": "center"}),
dcc.Markdown("""
Thus, the goal is to maximize the ELBO with respect to $$\\phi = \\{\\mu_\\theta, \\sigma_\\theta, \\mu_\\beta, \\sigma_\\beta,\\mu_\\eta, \\sigma_\\eta, \\mu_x, \\sigma_x\\}$$.

The most important is the initializations of the variational parameters $$\\phi$$ and their respective variational distributions.""", mathjax=True),
dbc.Row([dcc.Markdown(""" 
                        `loc`: location variables $\\mu$
                        `scale`: scale variables $\\sigma$
                        $\\mu_\\eta$: `ideological_topic_loc`
                        $\\sigma_\\eta$: `ideological_topic_scale`
    """, mathjax=True)],style={"text-align": "center"}),
dcc.Markdown("""The corresponding variational distribution is `ideological_topic_distribution`.  Below summarizes the above formulas in plainer language. """),
dbc.Row(children=[dbc.Col(children=["""1. Mean-Field Variational Family:""",html.Code("""
In mean-field variational inference, it is assumed that the posterior distribution factorizes across latent variables. This means that each latent variable is assumed to be independent of the others given certain parameters. For LDA, these parameters might include the distribution of topics in documents and the distribution of words in topics.
""")]),dbc.Col(children=["""2. Init and Optimization Process:""",html.Code("""
Start with some initial parameters for the variational family. Optimize these parameters to make the approximating distribution as close as possible to the true posterior distribution.
In this case, it's through maximizing the evdience lower bound or minimizing the KL-Divergence between these two probability distributions.
""")]),dbc.Col(children=["""3. Projecting Latent Information:""",html.Code("""
The parameters of the variational family can be interpreted as the estimated distributions of topics. 
These distributions are used to project latent information onto the documents and words. 
Each document gets a distribution over topics, and each topic gets a distribution over words.""")]),dbc.Col(children=["""4. Discovering Latent Topics:""",html.Code("""
The algorithm, through this variational inference process, discovers latent topics in the corpora based on how words co-occur across documents. 
Alternatively, it's measuring the pointwise mutual information between two probability distributions.
""")])]),
"""
The default corpus for this Colab notebook is """ ,html.A("Senate speeches", href="https://data.stanford.edu/congress_text"), """ from the 114th Senate session (2015-2017). The project also used the following corpora: Tweets from 2022 Democratic presidential candidates.""",]),
"""To replicate the whole process with my own Twitter data, I followed the steps below:""",
dcc.Markdown("""

        * `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
        * `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
        * `vocabulary.txt`: a `[num_words]` - length file where each line denotes the corresponding word in the vocabulary.
        * `author_map.txt`: a `[num_authors]` - length file where each line denotes the name of an author in the corpus.

""",mathjax=True),dcc.Markdown("""Please checkout this [notebook](https://colab.research.google.com/github/pyro-ppl/numpyro/blob/5291d0627d68598cf78b8ea97c540268660925c1/notebooks/source/tbip.ipynb) for the full implementation in Python"""), html.Br(), html.H5("""Resulting Ideological Distribution Generated from Author's Political Tweet"""), html.Hr()
],className='page2',style=PAGE_STYLE)

@callback(
    Output('cand-names-col-p2', 'children'),
    [dash.dependencies.Input('state-dropdown-p2', 'value')])
def update_output(value):
    a = states.loc[states['name'] == value, 'state']

    dropdown = []
    if value:
        res = candidates.loc[candidates['State'] == a.iloc[0], 'Candidate name'].tolist()
        dropdown = [dcc.Dropdown(['Select all'] + res,id='names-dropdown-p2', value=res[0], searchable=True,  multi=True)]

    else:
        res = candidates['Candidate name'].tolist()
        dropdown = [dcc.Dropdown(res ,id='names-dropdown-p2', value=res[0], searchable=True,  multi=True)]
    return [html.Label(['Select Candidate'], style={'font-size': '13px', "text-align": "left", "off-set":4, "color": "#808080"})] + dropdown






@callback(
    Output('bar-graph-plotly', 'figure'),
    [
        dash.dependencies.Input('bar-graph-plotly', 'figure')
        # whatever other inputs
    ]
)
def my_callback(figure_empty):

    fig = px.scatter(authors, x=['ideal_point'], y=[1]*len(authors),hover_data=['name'])
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    x1 = authors[authors['ideal_point'] > 0]
    x2 = authors[authors['ideal_point'] < 0]
    y = np.array([1] * len(authors))

    layout = go.Layout(
        xaxis={'title': 'Author\'s Ideal Point from Moderate to Progressive',
                        'visible': True,
                        'showticklabels': True},
        yaxis={'title': 'y-label',
                        'visible': False,
                        'showticklabels': False},

        height=200,
        margin=dict(l=0, r=0, t=0, b=0)
        )

    fig = go.Figure(layout=layout)


    fig.add_trace(go.Scatter(x=np.array(x1['ideal_point']),
                            y=y,
                            mode='markers',
                            name='progressive',
                            marker=dict(color='blue'), hovertemplate='<br><b>Ideal Point</b>: %{x}<br>'+'<b>%{text}</b>', text = ['Author: {}'.format(i) for i in x1['name'].tolist()],showlegend=False))
    fig.add_trace(go.Scatter(x=np.array(x2['ideal_point']),
                            y=y,
                            mode='markers',
                            name='conservative',
                            marker=dict(color='red'), hovertemplate='<br><b>Ideal Point</b>: %{x}<br>'+'<b>%{text}</b>', text = ['Author: {}'.format(i) for i in x2['name'].tolist()], showlegend=False)
                            )

    return fig



@callback(
    Output(component_id='graph-matplotlib', component_property='src'),
    [
        dash.dependencies.Input('state-dropdown-p2', 'value'), dash.dependencies.Input('names-dropdown-p2', 'value')
        # whatever other inputs
    ]
)
def my_callback(state_choice, pillar_dropdown):
    a = states.loc[states['name'] == state_choice, 'state']

    selected_authors = []
    if pillar_dropdown:
        if set(pillar_dropdown) == 'Select all':
            selected_authors = candidates.loc[candidates['State'] == a.iloc[0], 'Candidate name'].tolist()
        else:
            selected_authors = pillar_dropdown

    return pillar_dropdown
