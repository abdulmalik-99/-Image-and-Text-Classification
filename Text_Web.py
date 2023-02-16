import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output,State
import joblib

import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.express as px
import string
import pyarabic.araby as araby
import pyarabic.trans
from pyarabic.araby import strip_tashkeel ,normalize_ligature
import re 
from nltk.corpus import stopwords
from nltk.stem import arlstem
from nltk.stem.isri import ISRIStemmer
from nltk.stem.arlstem2 import ARLSTem2


# Load the saved models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
random_forest_classifier = joblib.load('random_forest_classifier_final_model.joblib')






arabic_punctuations='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟”“``….'+'0123456789'

def Remove_punctuations(txt):
    return txt.translate(str.maketrans('', '', arabic_punctuations))

def Text_Cleaning(text):
    text=pyarabic.trans.normalize_digits(text, source='all', out='west') # convert diffrent type of numbers into E number
    text = re.sub("[a-zA-Z]", "", text) # remove english letters
    # text = re.sub('\n', '', text) # remove \n from text
    text = re.sub(r"\s+", ' ',text) #remove long spaces 
    # text = re.sub(r'\d+', ' ', text) #remove number
    # text = re.sub(r'http\S+', '', text) # remove links
    text = text.translate(str.maketrans('','', arabic_punctuations)) # remove punctuation
    text = re.sub(' +', ' ',text) # remove extra space
    return text


def normlize(text):
    text = re.sub("ة", "ه", text)
    text = re.sub("[إأآا]", "ا", text)
    text=araby.normalize_hamza(text)
    text=normalize_ligature(text)
    text=strip_tashkeel(text)
    return text

Arabic_stopwords=[]
for item in stopwords.words('arabic'):
    Arabic_stopwords.append(normlize(item))


def Tokens_Stop_Words(text):
    tokens = araby.tokenize(text)
    return [w for w in tokens if w not in Arabic_stopwords ]

from nltk.stem.isri import ISRIStemmer
stemmer = ARLSTem2()
ISRIS_st = ISRIStemmer()
def stem(arr):
    arr=[stemmer.stem(word) for word in arr]
    return ' '.join(arr)


def predection_preprocessing(txt):
    txt=pd.Series(data=txt)
    txt=txt.apply(Text_Cleaning).apply(Remove_punctuations).apply(normlize).apply(Tokens_Stop_Words).apply(stem)
    return txt


# initialize the Dash app
app = dash.Dash(__name__)

# define the layout of the app
app.layout = html.Div([
    html.H1('Text Classification Web App'),
    html.Div([
        dcc.Textarea(
            id='text-input',
            value='Enter your text here...',
            style={'width': '100%', 'height': 300}
        ),
        html.Button('Submit', id='button'),
        html.Div(id='output-text')
    ])
])

# define the callback function that runs when the Submit button is clicked
@app.callback(
    Output(component_id='output-text', component_property='children'),
    [Input(component_id='button', component_property='n_clicks')],
    [State(component_id='text-input', component_property='value')]
)
def predict_text(n_clicks, input_text):
    # Check if input text is empty
    if  input_text=='Enter your text here...' or input_text=='' :
        return 'no prediction'

    # Preprocess the input text
    input_text = [input_text]
    input_text=predection_preprocessing(input_text)
    input_features = tfidf_vectorizer.transform(input_text)

    # Make the prediction
    predicted_class = random_forest_classifier.predict(input_features)[0]

    # Return the predicted class
    return f'The predicted class is {predicted_class}'
# run the app
if __name__ == '__main__':
    app.run_server(debug=False,port=8085)