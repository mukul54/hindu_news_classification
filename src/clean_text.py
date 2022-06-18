import json, os, re
import pickle, time
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def remove_stop_words(data):
  '''
  Remove stopwords
  '''
  word_list = word_tokenize(data)
  stops = set(stopwords.words("english"))
  filtered_words = [word for word in word_list if word not in stops]
  return " ".join(filtered_words)

def lemmatize_doc(doc):
  """
  Lemmatize a doc
  """
  lemma_sentence=[]
  wnl = WordNetLemmatizer()
  for word, tag in pos_tag(word_tokenize(doc)):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
      lemma = word
    else:
      lemma = wnl.lemmatize(word, wntag)
    lemma_sentence.append(lemma)
    lemma_sentence.append(" ")
  return "".join(lemma_sentence)


def combine_labels(df):
  '''
  Combine labels
  '''
  df['labels']=df.labels.replace(['business','economy','markets','cricket', 'races','other sports', 'entertainment', 'movies'], ['business','business','business', 'sports', 'sports','sports', 'entertainment', 'entertainment'])
  df['labels'] = df.labels.replace(['science','technology'],['sci-tech', 'sci-tech'])
  # drop travel row since it has really less data points
  df = df[df.labels != 'travel']
  return df

def clean(data):
  """
  Args:
    data: single sentence
  Returns:
  cleaned data
  """
  data = re.sub(r'(?<={)[^}]*', '', data)
  #remove function name
  data = re.sub('([\w]+ getPlayerID)', '', data)
  data = re.sub('loadAsyncURL.*', '', data)
  #remove unicodes
  data = re.sub(r'[^\x00-\x7f]',r'', data)
  data = data.replace('\x0c', '')
  data = data.replace('\x0d', '')
  data = data.replace('\f', '')
  data = data.replace('\n', ' ')

  # removing mentions 
  data = re.sub("@\S+", "", data)
  # remove market tickers
  data = re.sub("\$", "", data)
  # remove urls
  data = re.sub("https?:\/\/.*[\r\n]*", "", data)

  # removing hashtags 
  data = re.sub("#", "", data)

  # Remove ticks and the next character
  #Notley's tactful -> Notley tactful
  data = re.sub("\'\w+", '', data)

  #Remove Numbers
  data = re.sub(r'\w*\d+\w*', '', data)

  # remove all other symbols numbers and white spaces
  char_safe = ['.', '?', '!',]
  data = "".join([character if (character.isalnum() or character in char_safe) else " " for character in data])
  #remove ., ?, !
  data = data.replace('.', ' ')
  data = data.replace('?', ' ')
  data = data.replace('!', ' ')
  # remove extra white spce in the middle
  data = re.sub(" +", " ", data)
  #remove extra spaces in the beginning
  data = data.strip()

  #lowercase the text
  data = data.lower()
  return data

def label_encode(df):
  labels_code = {'business':0, 
                'entertainment':1,
                'international':2,
                'national':3,
                'sci-tech':4,
                'society':5,
                'sports':6
                }
  # numeric labels
  df['numeric_labels'] = df['labels']
  df = df.replace({'numeric_labels':labels_code})
  return df

def save_cleaned_df(df):
  '''
  Save cleaned data
  '''
  data_dir = '../assets/clean_data'

  #data with stop words and no lemmatization
  df_cleaned = df[['headline','links','cleaned','labels','numeric_labels']]
  data_path = os.path.join(data_dir, 'cleaned_articles.csv')
  df_cleaned.to_csv(data_path, encoding='utf-8', index=False)

  #data without stop words and no lemmatization
  df_cleaned_sw = df[['headline','links','cleaned_no_sw','labels', 'numeric_labels']]
  data_path = os.path.join(data_dir, 'cleaned_article_no_sw.csv')
  df_cleaned_sw.to_csv(data_path, encoding='utf-8', index=False)

  #data without stop words but with lemmatization
  df_cleaned_lemma = df[['headline','links', 'cleaned_no_sw_lemma','numeric_labels']]
  data_path = os.path.join(data_dir, 'cleaned_article_no_sw_lemma.csv')
  df_cleaned_lemma.to_csv(data_path, encoding='utf-8', index=False)

if __name__ == '__main__':
  raw_data_path = '../assets/raw_data/the_hindu_news_2020_60_days.csv'
  df = pd.read_csv(raw_data_path)
  df_new = combine_labels(df)
  print('...Cleaning Text:')
  df_new['cleaned'] = df_new.content.apply(clean)

  print('...Removing Stopwords:')
  df_new['cleaned_no_sw'] = df_new['cleaned'].apply(remove_stop_words)

  print('...Lemmatizing texts:')
  df_new['cleaned_no_sw_lemma'] = df_new['cleaned_no_sw'].apply(lemmatize_doc)
  df_new = label_encode(df_new)
  save_cleaned_df(df_new)
