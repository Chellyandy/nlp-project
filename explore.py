import pandas as pd
import numpy as np

#see the data
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

#play with words
import nltk.sentiment
import nltk
import re
#split and model
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split

#import 
from sklearn.feature_extraction.text import CountVectorizer

#sql creds
import env as e
import acquire as a


import os
import json

def split_data(df, variable):
    '''
    take in a DataFrame and target variable. return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.20, random_state=123, stratify=df[variable])
    
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123,
                                      stratify = train_validate[variable])
    return train, validate, test
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

def create_pie_chart(df, column_name,title):
    """ This function creates a pie chart for our categorical target variable"""
    values = df[column_name].value_counts()
    labels = values.index.tolist()
    sizes = values.tolist()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(title)
    plt.show()

def question_1(train):
    """
    Analyzes word frequency in different programming languages based on their cleaned contents.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    word_counts (pandas.DataFrame): DataFrame with word counts for each programming language and overall.

    """
    # we can do that process with a join on a Series and not just a list
    # we will do that all words and categories
    # we will pass our basic cleaning on top of that
    JavaScript_words = clean(' '.join(train[train.language=='JavaScript']['clean_contents']))
    Python_words = clean(' '.join(train[train.language=='Python']['clean_contents']))
    Java_words = clean(' '.join(train[train.language=='Java']['clean_contents']))
    TypeScript_words = clean(' '.join(train[train.language=='TypeScript']['clean_contents']))
    HTML_words = clean(' '.join(train[train.language=='HTML']['clean_contents']))
    Other_words = clean(' '.join(train[train.language=='Other']['clean_contents']))
    all_words = clean(' '.join(train['clean_contents']))

    # let's get some sights on word frequency by taking our words back apart
    # we will split each set of words by the spaces,
    # turn that into a list, cast that list as a Series,
    # and then take the value counts of that Series
    # We will do this for each type of word present

    JavaScript_words_freq = pd.Series(JavaScript_words).value_counts()
    Python_words_freq = pd.Series(Python_words).value_counts()
    Java_words_freq = pd.Series(Java_words).value_counts()
    TypeScript_words_freq = pd.Series(TypeScript_words).value_counts()
    HTML_words_freq = pd.Series(HTML_words).value_counts()
    Other_words_freq = pd.Series(Other_words).value_counts()
    all_words_freq = pd.Series(all_words).value_counts()

    #bring the above together
    word_counts = pd.concat([JavaScript_words_freq, Python_words_freq, Java_words_freq, TypeScript_words_freq, HTML_words_freq, Other_words_freq, all_words_freq], axis=1).fillna(0).astype(int)

    # rename the col names
    word_counts.columns = ['JavaScript','Python','Java', 'TypeScript', 'HTML', 'Other','All']

    #sort colums based on all
    word_counts.sort_values('All', ascending=False).head(10)

    #visualize
    word_counts.sort_values('All', ascending=False)[['JavaScript','Python','Java','TypeScript', 'HTML']].head(20).plot.barh()
    plt.xlabel('Word Count')
    plt.title('Top Word Count by language')
    plt.show()

    return word_counts.head()

def question_2(train):
    """
    Analyzes word frequency in specific libraries/tools across different programming languages.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    word_counts (pandas.DataFrame): DataFrame with word counts for each programming language, overall, and specific libraries/tools.

    """
    # we can do that process with a join on a Series and not just a list
    # we will do that all words and categories
    # we will pass our basic cleaning on top of that
    JavaScript_words = clean(' '.join(train[train.language=='JavaScript']['clean_contents']))
    Python_words = clean(' '.join(train[train.language=='Python']['clean_contents']))
    Java_words = clean(' '.join(train[train.language=='Java']['clean_contents']))
    TypeScript_words = clean(' '.join(train[train.language=='TypeScript']['clean_contents']))
    HTML_words = clean(' '.join(train[train.language=='HTML']['clean_contents']))
    Other_words = clean(' '.join(train[train.language=='Other']['clean_contents']))
    all_words = clean(' '.join(train['clean_contents']))

    # let's get some sights on word frequency by taking our words back apart
    # we will split each set of words by the spaces,
    # turn that into a list, cast that list as a Series,
    # and then take the value counts of that Series
    # We will do this for each type of word present

    JavaScript_words_freq = pd.Series(JavaScript_words).value_counts()
    Python_words_freq = pd.Series(Python_words).value_counts()
    Java_words_freq = pd.Series(Java_words).value_counts()
    TypeScript_words_freq = pd.Series(TypeScript_words).value_counts()
    HTML_words_freq = pd.Series(HTML_words).value_counts()
    Other_words_freq = pd.Series(Other_words).value_counts()
    all_words_freq = pd.Series(all_words).value_counts()

    #filter by specific library
    A_filtered_list = [word for word in all_words if word in ['flexbox','chatgpt','cli','stackblitz','angular','apm',
                                                              'opencv','zendesk', 'bootstrap', 'jquery','virtualbox',
                                                              'vagrant','nbsp','machinelearning', 'apache','dubbo', 
                                                              'alibaba','pandas','numpy']]
    filtered_list_freq=pd.Series(A_filtered_list).value_counts()

    word_counts = pd.concat([JavaScript_words_freq, Python_words_freq, Java_words_freq,TypeScript_words_freq,HTML_words_freq, Other_words_freq,all_words_freq,filtered_list_freq], axis=1).fillna(0).astype(int)

    # rename the col names
    word_counts.columns = ['JavaScript','Python','Java','TypeScript','HTML', 'Other','All','tools_frameworks']

    # visualize
    word_counts.sort_values('tools_frameworks', ascending=False)[['JavaScript','Python','Java','TypeScript','HTML']].head(14).plot.barh()
    plt.xlabel('Word Count')
    plt.title('Word Count by language specific library')
    plt.show()

    return word_counts.head()

def question_3(train):
    """
    Determines the most used words and their corresponding language across different programming languages.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    most_used_words_per_column (pandas.Series): Series containing the most used words and their corresponding language.

    """
    # we can do that process with a join on a Series and not just a list
    # we will do that all words and categories
    # we will pass our basic cleaning on top of that
    JavaScript_words = clean(' '.join(train[train.language=='JavaScript']['clean_contents']))
    Python_words = clean(' '.join(train[train.language=='Python']['clean_contents']))
    Java_words = clean(' '.join(train[train.language=='Java']['clean_contents']))
    TypeScript_words = clean(' '.join(train[train.language=='TypeScript']['clean_contents']))
    HTML_words = clean(' '.join(train[train.language=='HTML']['clean_contents']))
    Other_words = clean(' '.join(train[train.language=='Other']['clean_contents']))
    all_words = clean(' '.join(train['clean_contents']))

    # let's get some sights on word frequency by taking our words back apart
    # we will split each set of words by the spaces,
    # turn that into a list, cast that list as a Series,
    # and then take the value counts of that Series
    # We will do this for each type of word present

    JavaScript_words_freq = pd.Series(JavaScript_words).value_counts()
    Python_words_freq = pd.Series(Python_words).value_counts()
    Java_words_freq = pd.Series(Java_words).value_counts()
    TypeScript_words_freq = pd.Series(TypeScript_words).value_counts()
    HTML_words_freq = pd.Series(HTML_words).value_counts()
    Other_words_freq = pd.Series(Other_words).value_counts()
    all_words_freq = pd.Series(all_words).value_counts()

    #bring the above together
    word_counts = pd.concat([JavaScript_words_freq, Python_words_freq, Java_words_freq, TypeScript_words_freq, HTML_words_freq, Other_words_freq, all_words_freq], axis=1).fillna(0).astype(int)

    # rename the col names
    word_counts.columns = ['JavaScript','Python','Java', 'TypeScript', 'HTML', 'Other','All']

    #sort colums based on all
    word_counts.sort_values('All', ascending=False).head(10)

    # Calculate the total count of words across all columns
    word_counts['Total'] = word_counts.sum(axis=1)

    # Sort the dataframe based on the 'Total' column in descending order
    word_counts_sorted = word_counts.sort_values('Total', ascending=False)

    # Extract the most used words
    most_used_words = word_counts_sorted.index[:10]  # Change the number as per your requirement

    pd.Series(most_used_words)

    most_used_words_per_column = word_counts.idxmax()
    
    return print(most_used_words_per_column)

def question_4(train):
    """
    Determines the least used words and their corresponding language across different programming languages.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    least_used_words_per_column (pandas.Series): Series containing the least used words and their corresponding language.

    """
    # we can do that process with a join on a Series and not just a list
    # we will do that all words and categories
    # we will pass our basic cleaning on top of that
    JavaScript_words = clean(' '.join(train[train.language=='JavaScript']['clean_contents']))
    Python_words = clean(' '.join(train[train.language=='Python']['clean_contents']))
    Java_words = clean(' '.join(train[train.language=='Java']['clean_contents']))
    TypeScript_words = clean(' '.join(train[train.language=='TypeScript']['clean_contents']))
    HTML_words = clean(' '.join(train[train.language=='HTML']['clean_contents']))
    Other_words = clean(' '.join(train[train.language=='Other']['clean_contents']))
    all_words = clean(' '.join(train['clean_contents']))

    # let's get some sights on word frequency by taking our words back apart
    # we will split each set of words by the spaces,
    # turn that into a list, cast that list as a Series,
    # and then take the value counts of that Series
    # We will do this for each type of word present

    JavaScript_words_freq = pd.Series(JavaScript_words).value_counts()
    Python_words_freq = pd.Series(Python_words).value_counts()
    Java_words_freq = pd.Series(Java_words).value_counts()
    TypeScript_words_freq = pd.Series(TypeScript_words).value_counts()
    HTML_words_freq = pd.Series(HTML_words).value_counts()
    Other_words_freq = pd.Series(Other_words).value_counts()
    all_words_freq = pd.Series(all_words).value_counts()

    #bring the above together
    word_counts = pd.concat([JavaScript_words_freq, Python_words_freq, Java_words_freq, TypeScript_words_freq, HTML_words_freq, Other_words_freq, all_words_freq], axis=1).fillna(0).astype(int)

    # rename the col names
    word_counts.columns = ['JavaScript','Python','Java', 'TypeScript', 'HTML', 'Other','All']

    #sort colums based on all
    word_counts.sort_values('All', ascending=False).head(10)

    # Calculate the total count of words across all columns
    word_counts['Total'] = word_counts.sum(axis=1)

    # Sort the dataframe based on the 'Total' column in descending order
    word_counts_sorted = word_counts.sort_values('Total', ascending=False)

    # Extract the most used words
    most_used_words = word_counts_sorted.index[:10]  # Change the number as per your requirement

    pd.Series(most_used_words)

    least_used_words_per_column = word_counts.idxmin()
    
    return print(least_used_words_per_column)

def stats_ANOVA_viz(train):
    """
    Performs ANOVA test and visualizes the relationship between message length and language category.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    None

    """

    # prepare categories for ANOVA test
    javascript_stats_ml = train[train.language=='JavaScript']['message_length']
    python_stats_ml = train[train.language=='Python']['message_length']
    java_stats_ml = train[train.language=='Java']['message_length']
    typescript_stats_ml = train[train.language=='TypeScript']['message_length']
    HTML_stats_ml = train[train.language=='HTML']['message_length']
    other_stats_ml = train[train.language=='Other']['message_length']

    # Perform the ANOVA test
    f_value, p_value = f_oneway(
                        javascript_stats_ml,
                        python_stats_ml,
                        java_stats_ml,
                        typescript_stats_ml,
                        HTML_stats_ml,
                        other_stats_ml)

    # Print the ANOVA test results
    print("F-value:", f_value)
    print("p-value:", p_value)
    
    #visualize
    sns.barplot(data=train, x='message_length', y='language')
    plt.xlabel('Message length')
    plt.ylabel('Language')
    plt.title('Message length vs. language category')
    plt.show()

    return

def word_cloud_all(train):
    """
    Generates a word cloud visualization of the most common words in the entire dataset.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    None

    """
    #set all words
    all_words = clean(' '.join(train['clean_contents']))
    #generate cloud
    img = WordCloud(background_color='white').generate(' '.join(all_words))
    #set parameters
    plt.imshow(img)
    plt.axis('off')
    plt.title('Most Common all Words')
    plt.show()

def sentiment(train):
    """
    Calculates and visualizes the distribution of sentiment across different programming languages in the dataset.

    Args:
    train (pandas.DataFrame): DataFrame containing the training data.

    Returns:
    pandas.DataFrame: DataFrame with the mean and median sentiment values for each programming language.

    """

    info = train.groupby('language').sentiment.agg(['mean','median'])

    # is the distribution for sentiment 

    sns.kdeplot(train[train.language=='JavaScript'].sentiment, label='JavaScript')
    sns.kdeplot(train[train.language=='Python'].sentiment, label='Python')
    sns.kdeplot(train[train.language=='Java'].sentiment, label='Java')
    sns.kdeplot(train[train.language=='TypeScript'].sentiment, label='TypeScript')
    sns.kdeplot(train[train.language=='HTML'].sentiment, label='HTML')
    sns.kdeplot(train[train.language=='Other'].sentiment, label='Other')


    plt.legend()
    plt.title("Distribution for sentiment of JavaScript vs. Python vs. Java vs. Other")
    plt.show()

    return info