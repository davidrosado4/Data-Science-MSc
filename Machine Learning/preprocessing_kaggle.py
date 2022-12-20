# -*- coding: utf-8 -*-
"""
This script is build to preprocess the jobs dataset according to 
conclusions extracted from the data exploration notebook.
"""

# Import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

import re

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

# Import train dataset
df_train = pd.read_csv('./input/train.csv', index_col = 'Id')

#df_train = df_train.head()

# Import test dataset
df_test = pd.read_csv('./input/test.csv', index_col = 'Id')

#df_test = df_test.head()


# Location split
def loc_split(df):
    df['location'] = df['location'].replace(np.NAN, 'nan')
    df['country'] = np.NAN
    df['region'] = np.NAN
    df['city'] = np.NAN
    
    for i in range(df.shape[0]):
        df['location'].iloc[i] = df['location'].iloc[i].replace(" ", "").lower()
        chunk = df['location'].iloc[i].split(',')
        for j in range(len(chunk)):
            if j == 0:
                df['country'].iloc[i] = chunk[0]
            elif j == 1:
                df['region'].iloc[i] = chunk[1]
            elif j == 2:
                df['city'].iloc[i] = chunk[2]
            
    df.drop('location', axis = 1, inplace = True)
    
    # Replace np.NAN as a new category
    for feat in ['country', 'region', 'city']:
        df[feat] = df[feat].replace(np.NAN, 'nan')
    
    return df


# Removing features
def feat_rm(df, feats):
    for feat in feats:
        df.drop(feat, axis = 1, inplace = True)
    
    return df
    

# Scaling features
def feat_scaled(df, feat):
    feat_scaled = df[feat].copy()
    
    feat_scaled = (feat_scaled - feat_scaled.mean()) / feat_scaled.std()
    
    df[feat] = feat_scaled
    
    return df


# Fill missing values
def fill_na(df, feats):
    for feat in feats:
        df[feat] = df[feat].replace(np.NAN, 'nan')
    return df


df_train = feat_scaled(loc_split(feat_rm(df_train.copy(), ['department', 'salary_range', 'job_id'])), 'required_doughnuts_comsumption')
df_test = feat_scaled(loc_split(feat_rm(df_test.copy(), ['department', 'salary_range', 'job_id'])), 'doughnuts_comsumption')


TARGET = df_train['fraudulent']
df_train = feat_rm(df_train, ['fraudulent'])


print('First step: DONE')

# Feature Encoding
# Raise flag if company_profile is nan
df_train['company_profile_flag'] = df_train['company_profile'].isnull().astype(int)
df_test['company_profile_flag'] = df_test['company_profile'].isnull().astype(int)


# Hashing-----------------------------------------------------
hash_feats = ['country', 'region', 'city']

# Hash function
def StringHash(a, m=750, C=1024):
# m represents the estimated cardinality of the items set
# C represents a number that is larger than ord(c)
    hashes = []
    for w in a:
        hash = 0
        for i in range(len(w)):
            hash = (hash * C + ord(w[i])) % m
        hashes.append(hash)
    return pd.Series(hashes)

# Hash implementation
def generate_hash(df, feats):
    df[feats] = df[feats].apply(StringHash)
    return df

df_train = generate_hash(df_train.copy(), hash_feats)
df_test = generate_hash(df_test.copy(), hash_feats)

print('Hash: DONE')




# One-Hot-----------------------------------------------------
onehot_feats = ['required_experience', 'required_education',
                'employment_type']

# Supress non useful categories (that is, there are not in testing and training at the same time)
def compare_and_suppress(df1, df2, feats):
  """Compare and suppress
  This function receives two dataframes and compares the labels in the features to_compare and suppresses with NaN
  It returns two datasets with NaN where label are not present in both o them
  """
  for feat in feats:
    for value in df1[feat].values:
      if (value in df2[feat].values) == False:
        df1.replace(value, np.nan)

    for value in df2[feat].values:
      if (value in df1[feat].values) == False:
        df2.replace(value, np.nan)

  return df1, df2

# One-Hot implementation
def generate_onehot(df, feats):
    return pd.concat([df, pd.get_dummies(df[feats], prefix = feats)], axis = 1, join = 'inner')

df_train[onehot_feats] = df_train[onehot_feats].fillna('Unknown')
df_test[onehot_feats] = df_test[onehot_feats].fillna('Unknown')

df_train, df_test = compare_and_suppress(df_train, df_test, onehot_feats)

df_train = generate_onehot(df_train.copy(), onehot_feats)
df_test = generate_onehot(df_test.copy(), onehot_feats)

df_train = feat_rm(df_train, onehot_feats)
df_test = feat_rm(df_test, onehot_feats)

print('One-Hot: DONE')



# Bag of words------------------------------------------------
bow_feats = ['industry', 'function', 'benefits',
            'requirements', 'company_profile',
            'title', 'description']


# Lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# Generate vocabulary
def generate_vocab(df, feats):
    # Replace digits, small words and punctuation
    for feats in bow_feats:
        vals = []
        for val in df[feats].values:
            vals.append(re.sub(r"(?<=\w)([A-Z])", r" \1", val))
        df[feats] = vals
        df[feats] = df[feats].str.replace('\d+', ' ') # for digits
        df[feats] = df[feats].str.replace(r'(\b\w{1,2}\b)', ' ') # for words
        df[feats] = df[feats].str.replace('[^\w\s]', ' ') # for punctuation 
        df[feats] = df[feats].str.replace('_', ' ') # for punctuation 
        
        
    allfeats = df[bow_feats[0]].str.cat(df[bow_feats[1:]], sep=' ')
    
    allsen = []
    
    for i in range(len(allfeats)):
        allsen.append(allfeats[i])
    
    # Define vector count
    vectorizer = CountVectorizer(stop_words = 'english', 
                                 tokenizer = LemmaTokenizer(),
                                 strip_accents = 'unicode')
    
    wordCount = vectorizer.fit_transform(allsen)
    
    # tf-idf implementation
    tfIdfTrans = TfidfTransformer(use_idf = True)
    tfIdf = tfIdfTrans.fit_transform(wordCount)
    
    new_df = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    new_df = new_df.sort_values('TF-IDF', ascending=False)
    
    # Relevant words from TF-IDF
    rel_words = new_df.index[new_df['TF-IDF'] > 0].tolist()
    
    return vectorizer.get_feature_names_out(), vectorizer, rel_words


# Perform bag of words in dataframe
def generate_bow(df, bow_feats, vocab, vectorizer, tfidf, do_pca = False):    
    # Replace digits, small words and punctuation
    for feat in bow_feats:
        vals = []
        for val in df[feat].values:
            vals.append(re.sub(r"(?<=\w)([A-Z])", r" \1", val))
        df[feat] = vals
        df[feat] = df[feat].str.replace('\d+', '') # for digits
        df[feat] = df[feat].str.replace(r'(\b\w{1,2}\b)', '') # for words
        df[feat] = df[feat].str.replace('[^\w\s]', ' ') # for punctuation 
        df[feat] = df[feat].str.replace('_', ' ') # for punctuation 
        
    allfeats = df[bow_feats[0]].str.cat(df[bow_feats[1:]], sep=' ')
    
    allsen = []
    
    for i in range(len(allfeats)):
        allsen.append(allfeats[i])
    
    # Generate bag of words
    bag = vectorizer.transform(allsen).toarray()
    bow = pd.DataFrame(bag, columns = vocab)
    
    # Keep only relevant words from TF-IDF
    if do_pca:
        pca = PCA(n_components = 200)
        pca.fit(bow)
        loadings = pd.DataFrame(pca.components_.T,
                                #columns=['PC%s' % _ for _ in range(len(bow.columns))],
                                #index=bow.columns
                                )
    else:
        loadings = bow.loc[:, tfidf]

    return loadings


print('Generating BOW...')
df_train = fill_na(df_train.copy(), bow_feats)
df_test = fill_na(df_test.copy(), bow_feats)

vocab, vectorizer, tfidf = generate_vocab(df_train, bow_feats)
print('Vocab: DONE')


print(len(vocab))


do_pca = False


bow_train = generate_bow(df_train, bow_feats, vocab, vectorizer, tfidf, do_pca)
print('BOW train: DONE')
bow_test = generate_bow(df_test, bow_feats, vocab, vectorizer, tfidf, do_pca)
print('BOW test: DONE')

df_train = feat_rm(df_train, bow_feats)
df_test = feat_rm(df_test, bow_feats)

df_train = pd.concat([df_train.copy(), bow_train], axis = 1, join = 'inner')
df_test = pd.concat([df_test.copy(), bow_test], axis = 1, join = 'inner')
print('BOW: DONE')


#------------------------------------------------------------

# Save preprocessing
if do_pca:
    TARGET.to_csv('pca_fraudulent.csv')
    df_train.to_csv('pca_clear_train.csv')
    df_test.to_csv('pca_clear_test.csv')
else:
    TARGET.to_csv('fraudulent.csv')
    df_train.to_csv('clear_train.csv')
    df_test.to_csv('clear_test.csv')


df_train.drop('required_doughnuts_comsumption', axis = 1, inplace = True)
df_test.drop('doughnuts_comsumption', axis = 1, inplace = True)

if do_pca:
    df_train.to_csv('pca_nd_clear_train.csv')
    df_test.to_csv('pca_nd_clear_test.csv')
else:
    df_train.to_csv('nd_clear_train.csv')
    df_test.to_csv('nd_clear_test.csv')

print('Save: DONE')



