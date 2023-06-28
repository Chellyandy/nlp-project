import pandas as pd
import numpy as np
import time

#see the data
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

#play with words
import nltk.sentiment
import nltk
import re
from pprint import pprint

#split and model
from scipy.stats import f_oneway
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import ToktokTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#import 
from sklearn.feature_extraction.text import CountVectorizer

#sql creds
import env as e
import acquire as a
#scraping
import requests
from bs4 import BeautifulSoup

import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# setting basic style parameters for matplotlib
plt.rc('figure', figsize=(13, 7))
plt.style.use('seaborn-darkgrid')

def tokenize(text):
    """
    Tokenizes the words in the input string.
    """
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def clean(text: str) -> list: 
    """A simple function to cleanup text data"""
    
    #remove non-ascii characters & lower
    text = (text.encode('ascii', 'ignore')
                .decode('utf-8', 'ignore')
                .lower())
    
    #remove special characters
    words = re.sub(r'[^\w\s]', ' ', text).split()
    
    #build the lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    #getting all stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    return [wnl.lemmatize(word) for word in words if word not in stopwords]

def nlp_wrangle():
    #get data
    df = pd.read_json('data2.json')

    #clean data
    df['clean_contents'] = df.readme_contents.apply(tokenize).apply(' '.join)
    df['clean_contents'] = df.clean_contents.apply(clean).apply(' '.join)

    # words to remove
    words_to_remove = ["http", "com", "124","www","1","github","top","go","android"
                       ,"content","table","107","markdown","0","1","2","3","4","5",
                       "6","7","8","9","md"]

    # Iterate over each word in the 'words_to_remove' list
    for word in words_to_remove:
        df['clean_contents'] = df['clean_contents'].str.replace(word, '')

    # create nltk.sentiment.SentimentIntensityAnalyser()
    sia = nltk.sentiment.SentimentIntensityAnalyzer()

    #apply to dataframe
    df['sentiment'] = df['clean_contents'].apply(lambda doc: sia.polarity_scores(doc)['compound'])
    
    # add two new columns 'message_length' and 'word_count'
    df['message_length'] = df['clean_contents'].str.len()

    # we apply our clean function, apply len chained on it
    df['word_count'] = df.clean_contents.apply(clean).apply(len)

    #call top 5 languages to keep and assign all others to other
    languages_to_keep = ['JavaScript', 'Python', 'Java', 'TypeScript', 'HTML']
    df['language'] = np.where(df['language'].isin(languages_to_keep), df['language'], 'Other')

    # Filter the DataFrame based on conditions for word_count and message_length
    df = df.loc[(df['word_count'] <= 10000) & (df['message_length'] <= 60000)]

    return df