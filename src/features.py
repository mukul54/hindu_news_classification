import json, os, re
import pickle
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit


def load_cleaned_data(removed_sw = False, lemmatized = False):
  data_dir = '/content/drive/MyDrive/news_analysis/assets/clean_data'
  if lemmatized:
    cleaned_data_path = os.path.join(data_dir, 'cleaned_article_no_sw_lemma.csv')
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_cleaned = df_cleaned.rename(columns={"cleaned_no_sw_lemma": "cleaned_article"})
  elif removed_sw:
    cleaned_data_path = os.path.join(data_dir, 'cleaned_article_no_sw.csv')
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_cleaned = df_cleaned.rename(columns={"cleaned_no_sw": "cleaned_article"})
  else:
    cleaned_data_path = os.path.join(data_dir, 'cleaned_articles.csv')
    df_cleaned = pd.read_csv(cleaned_data_path)
    df_cleaned = df_cleaned.rename(columns={"cleaned": "cleaned_article"})
    
  return df_cleaned



def split_stratified_into_train_val_test( df_input, 
                                          stratify_colname='y',
                                          frac_train=0.8,
                                          frac_val=0.1, 
                                          frac_test=0.1,
                                          random_state=None
                                        ):

    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def vectorize_text(df_cleaned, max_features = 300):
  #consider top max_features
  tfidf = TfidfVectorizer(max_features = max_features, ngram_range=(1,2))

  #train val test split
  train, val, test = split_stratified_into_train_val_test(df_cleaned, stratify_colname='numeric_labels')


  x_train = train['cleaned_article']
  x_val = val['cleaned_article']
  x_test = test['cleaned_article']

  y_train = train['numeric_labels']
  y_val = val['numeric_labels']
  y_test = test['numeric_labels']
  
  features_train = tfidf.fit_transform(x_train).toarray() 
  features_val =   tfidf.transform(x_val).toarray()#never again use tfidf.fit_transform on val or test set
  features_test = tfidf.transform(x_test).toarray()#never again use tfidf.fit_transform on val or test set
  
  pickles_dir = '../data_pickles'
  ###########################################################
  # x_train
  xtrain_p = os.path.join(pickles_dir, 'x_train.pickle')
  with open(xtrain_p, 'wb') as output:
    pickle.dump(x_train, output)
  # x_val
  xval_p = os.path.join(pickles_dir, 'x_val.pickle')
  with open(xval_p, 'wb') as output:
    pickle.dump(x_val, output)
  # x_test    
  xtest_p = os.path.join(pickles_dir, 'x_test.pickle')
  with open(xtest_p, 'wb') as output:
    pickle.dump(x_test, output)

  ########################################################
  # y_train
  ytrain_p = os.path.join(pickles_dir, 'y_train.pickle')
  with open(ytrain_p, 'wb') as output:
    pickle.dump(y_train, output)
  # y_val
  yval_p = os.path.join(pickles_dir, 'y_val.pickle')
  with open(yval_p, 'wb') as output:
    pickle.dump(y_val, output)
  # y_test
  ytest_p = os.path.join(pickles_dir, 'y_test.pickle')
  with open(ytest_p, 'wb') as output:
    pickle.dump(y_test, output)
  
  ###########################################################
  # features_train
  features_train_p = os.path.join(pickles_dir, 'features_train.pickle')
  with open(features_train_p, 'wb') as output:
    pickle.dump(features_train, output)
  # features_val
  features_val_p = os.path.join(pickles_dir, 'features_val.pickle')
  with open(features_val_p, 'wb') as output:
    pickle.dump(features_val, output)
  # features_test
  features_test_p = os.path.join(pickles_dir, 'features_test.pickle')
  with open(features_test_p, 'wb') as output:
    pickle.dump(features_test, output)

  ###########################################################
  # df_cleaned
  df_cleaned_p = os.path.join(pickles_dir, 'df_cleaned.pickle')
  with open(df_cleaned_p, 'wb') as output:
    pickle.dump(df_cleaned, output)
  # TF-IDF object
  tfidf_p = os.path.join(pickles_dir, 'tfidf.pickle')
  with open(tfidf_p, 'wb') as output:
    pickle.dump(tfidf, output)
  
  return features_train, y_train, features_val, y_val, features_test, y_test
if __name__ == '__main__':
  df_cleaned = load_cleaned_data(lemmatized = True)
  features_train, labels_train, features_val, labels_val, features_test, labels_test = vectorize_text(df_cleaned)