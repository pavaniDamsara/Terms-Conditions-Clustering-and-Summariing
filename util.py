import os
import re
import pandas as pd
import numpy as np
from variables import*
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

def get_csv_data(csv_path):
    '''
        Get each CSV seperately.Then filter columns and preprocess dataframe
    '''
    df = pd.read_csv(csv_path, encoding='ISO 8859-1')
    df.columns = map(str.lower, df.columns)
    df = df[['terms list', 'main category']]
    df['main category'] = df['main category'].str.lower()
    df['main category'] = df['main category'].str.strip()
    df = df.dropna(axis=1, how='all') 
    df = df[df['terms list'].notna()]
    df = df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.fillna(method='ffill')
    return df

def load_csvs():
    '''
        Load each preprocessed dataframe from get_csv_data function and concatenate into one long dataframe
    '''
    csv_files = os.listdir(csv_file_paths)
    for i,csv_file in enumerate(csv_files):
        csv_path = os.path.join(csv_file_paths, csv_file)
        df = get_csv_data(csv_path)
        if (i == 0):
            final_df = df 
        else:
            final_df = pd.concat([final_df, df], ignore_index=True)
    final_df = final_df.dropna(axis=0, how='any') 
    return final_df

def lemmatization(lemmatizer,sentence):
    '''
        Lematize texts in the terms list
    '''
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    '''
        Remove stop words in texts in the terms list
    '''
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(row):
    '''
        Text preprocess on term text using above functions
    '''
    term = row['terms list']
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    term = term.lower()
    remove_punc = tokenizer.tokenize(term) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_term = ' '.join(remove_stop)
    return updated_term

def preprocessed_data(terms):
    '''
        Preprocess entire terms list
    '''
    updated_terms = []
    if isinstance(terms, np.ndarray) or isinstance(terms, list):
        for term in terms:
            updated_term = preprocess_one(term)
            updated_terms.append(updated_term)
    elif isinstance(terms, np.str_)  or isinstance(terms, str):
        updated_terms = [preprocess_one(terms)]

    return np.array(updated_terms)
    
def load_data():
    '''
        Encode labels and then split into train and test data.
    '''
    if not os.path.exists(final_csv_path):
        df = load_csvs()
        df['preprocessed terms'] = df.apply(preprocess_one, axis=1)
        df.to_csv(final_csv_path, index=False)

    df = pd.read_csv(final_csv_path)
    df = shuffle(df)
    print(df.head())
    classes = df['main category'].values 
    encoder = LabelEncoder()
    encoder.fit(classes)
    classes = encoder.transform(classes)
    policy_texts = df['preprocessed terms'].values

    return policy_texts
    # Ntrain = int(cutoff * len(classes))
    # X, Y = shuffle(policy_texts, classes)
    # Xtrain, Xtest = X[:Ntrain], X[Ntrain:]
    # Ytrain, Ytest = Y[:Ntrain], Y[Ntrain:]
    # return Xtrain, Xtest, Ytrain, Ytest

load_data()